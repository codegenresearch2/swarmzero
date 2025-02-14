import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from langtrace_python_sdk import inject_additional_attributes  # type: ignore   # noqa
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from swarmzero.chat import ChatManager
from swarmzero.chat.schemas import ChatData, ChatHistorySchema
from swarmzero.database.database import DatabaseManager, get_db
from swarmzero.llms.openai import OpenAIMultiModalLLM
from swarmzero.sdk_context import SDKContext
from swarmzero.server.routes.files import insert_files_to_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm_instance(id, sdk_context: SDKContext):
    attributes = sdk_context.get_attributes(
        id, "llm", "agent_class", "tools", "instruction", "tool_retriever", "enable_multi_modal", "max_iterations"
    )
    if attributes['agent_class'] == OpenAIMultiModalLLM:
        llm_instance = attributes["agent_class"](
            attributes["llm"],
            attributes["tools"],
            attributes["instruction"],
            attributes["tool_retriever"],
            max_iterations=attributes["max_iterations"],
        ).agent
    else:
        llm_instance = attributes["agent_class"](
            attributes["llm"], attributes["tools"], attributes["instruction"], attributes["tool_retriever"]
        ).agent
    return llm_instance, attributes["enable_multi_modal"]

def setup_chat_routes(router: APIRouter, id, sdk_context: SDKContext):
    async def validate_chat_data(chat_data):
        if len(chat_data.messages) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided",
            )
        last_message = chat_data.messages.pop()
        if last_message.role != MessageRole.USER:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user",
            )
        return last_message, [ChatMessage(role=m.role, content=m.content) for m in chat_data.messages]

    def is_valid_image(file: UploadFile) -> bool:
        return Path(file.filename).suffix.lower() in ChatManager.allowed_image_extensions

    @router.post("/chat")
    async def chat(
        request: Request,
        user_id: str = Form(...),
        session_id: str = Form(...),
        chat_data: str = Form(...),
        files: List[UploadFile] = File([]),
        db: AsyncSession = Depends(get_db),
    ):
        try:
            chat_data_parsed = ChatData.model_validate_json(chat_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Chat data is malformed: {e.json()}",
            )

        try:
            stored_files = await insert_files_to_index(files, id, sdk_context)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error while inserting files: {str(e)}",
            )

        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)

        chat_manager = ChatManager(
            llm_instance, user_id=user_id, session_id=session_id, enable_multi_modal=enable_multi_modal
        )
        db_manager = DatabaseManager(db)

        last_message, _ = await validate_chat_data(chat_data_parsed)

        image_files = [file for file in stored_files if is_valid_image(file)]

        try:
            response = await inject_additional_attributes(
                lambda: chat_manager.generate_response(db_manager, last_message, image_files), {"user_id": user_id}
            )
            return response
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error while generating chat response: {str(e)}",
            )

    # rest of the code...

In this version of the code, I've updated the `is_valid_image` function to take a `UploadFile` object as input instead of a string. This allows us to validate the image file type more efficiently.\n\nI've also added try-except blocks around the file insertion and response generation to improve error handling. If any errors occur during these processes, an HTTPException with the appropriate status code and detail message will be raised.
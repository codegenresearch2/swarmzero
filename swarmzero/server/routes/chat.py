import os\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import List\nfrom fastapi import (\n    APIRouter, \n    Depends, \n    File, \n    Form, \n    HTTPException, \n    Query, \n    Request, \n    UploadFile, \n    status, \n)\nfrom langtrace_python_sdk import inject_additional_attributes \nfrom llama_index.core.llms import ChatMessage, MessageRole\nfrom pydantic import ValidationError\nfrom sqlalchemy.ext.asyncio import AsyncSession\nfrom swarmzero.chat import ChatManager\nfrom swarmzero.chat.schemas import ChatData, ChatHistorySchema\nfrom swarmzero.database.database import DatabaseManager, get_db\nfrom swarmzero.llms.openai import OpenAIMultiModalLLM\nfrom swarmzero.sdk_context import SDKContext\nfrom swarmzero.server.routes.files import insert_files_to_index\nALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}\n\n\ndef get_llm_instance(id, sdk_context: SDKContext):\n    attributes = sdk_context.get_attributes(\n        id,\n        'llm',\n        'agent_class',\n        'tools',\n        'instruction',\n        'tool_retriever',\n        'enable_multi_modal',\n        'max_iterations'\n    )\n    if attributes['agent_class'] == OpenAIMultiModalLLM:\n        llm_instance = attributes['agent_class'](\n            attributes['llm'],\n            attributes['tools'],\n            attributes['instruction'],\n            attributes['tool_retriever'],\n            max_iterations=attributes['max_iterations'],\n        ).agent\n    else:\n        llm_instance = attributes['agent_class'](\n            attributes['llm'],\n            attributes['tools'],\n            attributes['instruction'],\n            attributes['tool_retriever'],\n        ).agent\n    return llm_instance, attributes['enable_multi_modal']\n\n\ndef setup_chat_routes(router: APIRouter, id, sdk_context: SDKContext):\n    async def validate_chat_data(chat_data):\n        if len(chat_data.messages) == 0:\n            raise HTTPException(\n                status_code=status.HTTP_400_BAD_REQUEST,\n                detail='No messages provided',\n            )\n        last_message = chat_data.messages.pop()\n        if last_message.role != MessageRole.USER:\n            raise HTTPException(\n                status_code=status.HTTP_400_BAD_REQUEST,\n                detail='Last message must be from user',\n            )\n        return last_message, [ChatMessage(role=m.role, content=m.content) for m in chat_data.messages]\n\n    def is_valid_image(file_path: str) -> bool:\n        return Path(file_path).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS\n\n    @router.post('/chat'):\n    async def chat(\n        request: Request,\n        user_id: str = Form(...),\n        session_id: str = Form(...),\n        chat_data: str = Form(...),\n        files: List[UploadFile] = File([]),\n        db: AsyncSession = Depends(get_db),\n    ):\n        try:\n            chat_data_parsed = ChatData.model_validate_json(chat_data)\n        except ValidationError as e:\n            raise HTTPException(\n                status_code=status.HTTP_400_BAD_REQUEST,\n                detail=f'Chat data is malformed: {e.json()}',\n            )\n\n        stored_files = await insert_files_to_index(files, id, sdk_context)\n        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)\n\n        chat_manager = ChatManager(\n            llm_instance,\n            user_id=user_id,\n            session_id=session_id,\n            enable_multi_modal=enable_multi_modal\n        )\n        db_manager = DatabaseManager(db)\n\n        last_message, _ = await validate_chat_data(chat_data_parsed)\n\n        image_files = [file for file in stored_files if is_valid_image(file)]\n\n        return await inject_additional_attributes(\n            lambda: chat_manager.generate_response(db_manager, last_message, image_files), \n            {'user_id': user_id}\n        )\n\n    @router.get('/chat_history', response_model=List[ChatHistorySchema]):\n    async def get_chat_history(\n        user_id: str = Query(...),\n        session_id: str = Query(...),\n        db: AsyncSession = Depends(get_db),\n    ):\n        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)\n\n        chat_manager = ChatManager(\n            llm_instance,\n            user_id=user_id,\n            session_id=session_id\n        )\n        db_manager = DatabaseManager(db)\n        chat_history = await chat_manager.get_messages(db_manager)\n        if not chat_history:\n            raise HTTPException(\n                status_code=status.HTTP_404_NOT_FOUND,\n                detail='No chat history found for this user',\n            )\n\n        return [\n            ChatHistorySchema(\n                user_id=user_id,\n                session_id=session_id,\n                message=msg.content,\n                role=msg.role,\n                timestamp=str(datetime.now(timezone.utc)),\n            )\n            for msg in chat_history\n        ]\n\n    @router.get('/all_chats'):\n    async def get_all_chats(\n        user_id: str = Query(...),\n        db: AsyncSession = Depends(get_db),\n    ):\n        llm_instance, enable_multi_modal = get_llm_instance(id, sdk_context)\n\n        chat_manager = ChatManager(\n            llm_instance,\n            user_id=user_id,\n            session_id=''\n        )\n        db_manager = DatabaseManager(db)\n        all_chats = await chat_manager.get_all_chats_for_user(db_manager)\n\n        if not all_chats:\n            raise HTTPException(\n                status_code=status.HTTP_404_NOT_FOUND,\n                detail='No chats found for this user',\n            )\n\n        return all_chats\n
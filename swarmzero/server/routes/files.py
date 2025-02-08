import logging\\\nimport os\\\nfrom typing import List\\\\\nfrom dotenv import load_dotenv\\\\nfrom fastapi import APIRouter, File, HTTPException, UploadFile\\\\nfrom swarmzero.filestore import BASE_DIR, FileStore\\\\nfrom swarmzero.sdk_context import SDKContext\\\\nfrom swarmzero.tools.retriever.base_retrieve import IndexStore, RetrieverBase\\\\n\nload_dotenv()\\\\n\n# Set up logging\\\\\nlogging.basicConfig(level=logging.INFO)\\\\nlogger = logging.getLogger(__name__)\\\n\nALLOWED_FILE_TYPES = [\\\\
import logging\nimport os\nfrom typing import List\nfrom dotenv import load_dotenv\nfrom fastapi import APIRouter, File, HTTPException, UploadFile\n\nload_dotenv()\n\n# Set up logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\n# Define allowed file types\nALLOWED_FILE_TYPES = [\n    'application/json',\n    'text/csv',\n    'text/plain',\n    'application/pdf',\n    'image/jpeg',\n    'image/jpg',\n    'image/png',\n    'application/msword',\n    'application/vnd.ms-excel',\n    'text/markdown'\n]\n\n# Initialize FileStore and IndexStore\nBASE_DIR = 'test_files'\nfile_store = FileStore(BASE_DIR)\nindex_store = IndexStore.get_instance()\nUSE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'\n\n# Function to insert files into index\nasync def insert_files_to_index(files: List[UploadFile], id: str, sdk_context: SDKContext):\n    saved_files = []\n    for file in files:\n        if not file.content_type:\n            logger.warning(f'File {file.filename} has no content type.')\n            raise HTTPException(status_code=400, detail='File content type is missing.')\n\n        if file.content_type not in ALLOWED_FILE_TYPES:\n            logger.warning(f'Disallowed file type upload attempted: {file.content_type}')\n            raise HTTPException(status_code=400, detail=f'File type {file.content_type} is not allowed.')\n\n        try:\n            agent = sdk_context.get_resource(id)\n            filename = await file_store.save_file(file)\n            if USE_S3:\n                file_path = filename\n            else:\n                file_path = f'{BASE_DIR}/{filename}'\n            saved_files.append(file_path)\n\n            if USE_S3:\n                continue  # TODO: update retrivers to use S3\n\n            if 'BaseRetriever' in index_store.list_indexes():\n                index = index_store.get_index('BaseRetriever')\n                RetrieverBase().insert_documents(index, [file_path])\n                index_store.update_index('BaseRetriever', index)\n                index_store.insert_index_files('BaseRetriever', [filename])\n                logger.info('Inserting data to existing basic index')\n                logger.info(f'Index: {index_store.list_indexes()}')\n                agent.recreate_agent()\n                index_store.save_to_file()\n\n            else:\n                retriever = RetrieverBase()\n                index, file_names = retriever.create_basic_index([file_path])\n                index_store.add_index(retriever.name, index, file_names)\n                logger.info('Inserting data to new basic index')\n                logger.info(f'Index: {index_store.list_indexes()}')\n                agent.recreate_agent()\n                index_store.save_to_file()\n\n        except ValueError as e:\n            logger.error(f'Value error: {e}')\n            raise HTTPException(status_code=400, detail=str(e))\n        except IOError as e:\n            logger.error(f'I/O error: {e}')\n            raise HTTPException(status_code=500, detail=str(e))\n        finally:\n            await file.close()\n            logger.info(f'Closed file {file.filename}')\n\n    return saved_files\n\n# Setup routes\n\ndef setup_files_routes(router: APIRouter, id: str, sdk_context: SDKContext):\n    @router.post('/uploadfiles/')\n    async def create_upload_files(files: List[UploadFile] = File(...)):\n        saved_files = await insert_files_to_index(files, id, sdk_context)\n        logger.info(f'Uploaded files: {saved_files}')\n        return {'filenames': saved_files}\n\n    @router.get('/files/')\n    async def list_files():\n        try:\n            files = file_store.list_files()\n            logger.info('Listed files')\n            return {'files': files}\n        except IOError as e:\n            logger.error(f'I/O error: {e}')\n            raise HTTPException(status_code=500, detail=str(e))\n\n    @router.delete('/files/{filename}')\n    async def delete_file(filename: str):\n        try:\n            if file_store.delete_file(filename):\n                logger.info(f'Deleted file {filename}')\n                return {'message': f'File {filename} deleted successfully.'}\n            else:\n                logger.warning(f'File {filename} not found for deletion')\n                raise HTTPException(status_code=404, detail=f'File {filename} not found.')\n        except ValueError as e:\n            logger.error(f'Value error: {e}')\n            raise HTTPException(status_code=400, detail=str(e))\n        except IOError as e:\n            logger.error(f'I/O error: {e}')\n            raise HTTPException(status_code=500, detail=str(e))\n\n    @router.put('/files/{old_filename}/{new_filename}')\n    async def rename_file(old_filename: str, new_filename: str):\n        try:\n            if file_store.rename_file(old_filename, new_filename):\n                logger.info(f'Renamed file from {old_filename} to {new_filename}')\n                return {'message': f'File {old_filename} renamed to {new_filename} successfully.'}\n            else:\n                logger.warning(f'File {old_filename} not found for renaming')\n                raise HTTPException(status_code=404, detail=f'File {old_filename} not found.')\n        except ValueError as e:\n            logger.error(f'Value error: {e}')\n            raise HTTPException(status_code=400, detail=str(e))\n        except IOError as e:\n            logger.error(f'I/O error: {e}')\n            raise HTTPException(status_code=500, detail=str(e))\n
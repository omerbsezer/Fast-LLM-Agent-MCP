from utils.aws_clients import get_config, get_s3_client


def upload_file(file_obj, filename: str) -> str:
    cfg = get_config()
    key = filename
    get_s3_client().upload_fileobj(file_obj, cfg["docs_bucket"], key)
    return key

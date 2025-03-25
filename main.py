from fastapi import FastAPI, File, Form, UploadFile
from pydantic import Optional
import logging

from pydantic import ValidationError

app = FastAPI()

logger = logging.getLogger(__name__)

@app.post("/upload_doc")
def upload(
    research_file : UploadFile = File(...),
    bib_file : Optional[UploadFile] = File(...),
    domain : str = Form(...)
):
    if not research_file.filename.endswith('.pdf'):
        error_msg = "Uploaded Document is not a .pdf file"
        logger.error(error_msg)
        raise ValidationError(error_msg)

    if not bib_file.filename.endswith('.bib'):
        logger.warn("Bib file hasn't been uploaded, could not retrieve possible information further.")


    try :
        ## DOCUMENT PROCESSING - ALGO

        ## 1. TEXT SPLITTER TO LIST OF DOCUMENTS :
        ## - parse pdf to list of documents (temporary storage must be in docker temp file and not host)
        ## - add .bib file annotations to metadata of all documents in the list of documents generated

        ## 2. SQL INDEX MANAGER
        ## - ** volume map the db file storage (sqlite) **
        ## - should validate things

        ## 3. (Optional) Store Unique Metadata

        ## 4. POST TO VECTOR DB : (later)
        ## - later ideate to create new indices or something



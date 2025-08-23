import fitz
import easyocr

reader = easyocr.Reader(['fr', 'en'])

def extraire_texte_depuis_image(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return " ".join(result)





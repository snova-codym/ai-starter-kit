import os
import glob
from pdf2image import convert_from_path
from paddleocr import PaddleOCR,  PPStructure

ocr = PaddleOCR(use_angle_cls=False, lang='en') # need to run only once to download and load model into memory
layout_engine = PPStructure(recovery=False, layout=True, table=True, ocr=False, show_log=False) # need to run only once to download and load model into memory



class TableTools:

    def convert_pdf_to_images(data_directory) -> None:
            """
                this method converts a pdf file to a series of images of each page
            Args:
                pdf_file_path (str): pdf file path
                output_folder (str, optional): output directory. Defaults to 'data/extraction'.

            Returns:
                str: _description_ output directory
            """

            files = glob.glob(data_directory + "**/**",
                                    recursive=True)
            files = [file for file in files if file.endswith(".pdf")]

            for filename in files:
                print(f"Converting {filename} to a folder of images in the same location")

                images = convert_from_path(filename)
                output_folder = filename.split('.')[0] + "/images"
                os.makedirs(output_folder, exist_ok=True)
                for i, image in enumerate(images):
                    img_name=f"page_{i}.jpg"
                    output_path = f"{output_folder}/{img_name}"
                    image.save(output_path, 'JPEG')

    def structured_ocr(self, img_file_path, ocr=layout_engine):
        """
        This method performs an structure deetction, table transcription and OCR on a single image
        Args:
            img_file_path (str): image file path
            ocr (PaddleOCR, optional): PaddleStructure engine object. Defaults to layout_engine.
        Returns:
            str: output ocr object
        """
        img = cv2.imread(img_file_path)
        return ocr(img)


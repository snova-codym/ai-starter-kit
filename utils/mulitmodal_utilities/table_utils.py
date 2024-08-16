import os
import glob
from pdf2image import convert_from_path



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
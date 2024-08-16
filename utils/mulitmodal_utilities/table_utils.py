import os
import glob
from PIL import Image
from pdf2image import convert_from_path
from ultralyticsplus import YOLO
# from paddleocr import PaddleOCR,  PPStructure

# ocr = PaddleOCR(use_angle_cls=False, lang='en') # need to run only once to download and load model into memory
# layout_engine = PPStructure(recovery=False, layout=True, table=True, ocr=False, show_log=False) # need to run only once to download and load model into memory



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

    # def structured_ocr(self, img_file_path, ocr=layout_engine):
    #     """
    #     This method performs an structure deetction, table transcription and OCR on a single image
    #     Args:
    #         img_file_path (str): image file path
    #         ocr (PaddleOCR, optional): PaddleStructure engine object. Defaults to layout_engine.
    #     Returns:
    #         str: output ocr object
    #     """
    #     img = cv2.imread(img_file_path)
    #     return ocr(img)

    def crop_tables(data_directory: str, 
                    conf: float = 0.25,
                    iou: float = 0.45,
                    agnostic_nms: bool = False,
                    max_det: int = 1000,
                    threshold: float = 0.8,
                    offset: int = 20) -> None:
        
        """
        This method crops tables from images using YOLOv8
        Args:
            data_directory (str): directory of images
        """
        model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        model.overrides['conf'] = conf  # NMS confidence threshold
        model.overrides['iou'] = iou  # NMS IoU threshold
        model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        model.overrides['max_det'] = max_det  # maximum number of detections per image

        files = glob.glob(data_directory + "**/**", recursive=True)
        files = [file for file in files if file.endswith(".jpg")]

        for filename in files:
            print(f"Cropping tables from {filename}")
            results = model.predict(filename)
            boxes, mask = results[0].boxes.xyxy.numpy(), results[0].boxes.conf.numpy() >= threshold
            valid_boxes = boxes[mask]
            print(valid_boxes)
            loaded_img = Image.open(filename)
            output_folder = filename.partition('images')[0] + "cropped_tables/"
            os.makedirs(output_folder, exist_ok=True)
            for i, box in enumerate(valid_boxes):
                cropped_table = loaded_img.crop((box[0]-offset, 
                                                box[1]-offset, 
                                                box[2]+offset, 
                                                box[3]+offset))
                page_no = filename.partition('images')[-1].replace("/", "").replace(".jpg", "")
                output_path = f"{output_folder}/{page_no}_table_{i}.jpg"
                print(output_path)
                cropped_table.save(output_path, "JPEG")




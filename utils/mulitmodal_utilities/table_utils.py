import os
import cv2
import glob
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from pdf2image import convert_from_path
import random
import subprocess
import time
from typing import List
from ultralyticsplus import YOLO

logging.basicConfig(level=logging.INFO)

DOCUMENT_PREFIX_LANDSCAPE = r'''\documentclass[8pt]{article}
\usepackage[landscape, margin={MARGIN}in]{geometry} % Sets the document to landscape mode
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{colortbl} 

\usepackage{multirow}
\usepackage{booktabs}

\usepackage{FONT_TYPE}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\arrayrulewidth}{{RULE_WIDTH}mm}
\begin{document}
'''

DOCUMENT_PREFIX = r'''\documentclass[8pt]{article}
\usepackage[margin={MARGIN}in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{array}
\usepackage{colortbl} 

\usepackage{multirow}
\usepackage{booktabs}

\usepackage{FONT_TYPE}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\arrayrulewidth}{{RULE_WIDTH}mm}
\begin{document}
'''

font_types = [
    "lmodern",
    "times",
    "helvet", 
    "courier",
    "mathpazo",
    "newcent"
]


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
                logging.info(f"Converting {filename} to a folder of images in the same location")

                images = convert_from_path(filename)
                base_folder = filename.split('.')[0]
                output_folder = os.join(base_folder, "images")
                os.makedirs(output_folder, exist_ok=True)
                for i, image in enumerate(images):
                    img_name=f"page_{i}.jpg"
                    output_path = f"{output_folder}/{img_name}"
                    image.save(output_path, 'JPEG')

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
            
            boxes, mask = results[0].boxes.xyxy.numpy(), 
            results[0].boxes.conf.numpy() >= threshold
            
            valid_boxes = boxes[mask]
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
                cropped_table.save(output_path, "JPEG")

    def replace_special_to_latex(self, 
                                 text: str) -> str:
        
        text = text.replace("%", "\\%")
        text = text.replace("_", "\\_")
        text = text.replace("*", "\\quad ")
        text = text.replace("$", "\\$")

        return text
    
    def _return_latex_text(self, text: str):
        
        latex_text = ""

        for line in text.splitlines():
            line = line.replace("\t", " & ")
            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _return_colors(self) -> List[str]:

        colors = ["\\definecolor{mycolor}{RGB}{150,210,255}\n",
                  "\\definecolor{mycolor}{RGB}{210,210,210}\n",
                  "\\definecolor{mycolor}{RGB}{175,250,180}\n",
                  "\\definecolor{mycolor}{RGB}{250,250,180}\n",]
        
        return colors
    
    def _randomize_colors(self, text: str):

        r = np.random.uniform(0, 1)

        latex_text = ""

        for i ,line in enumerate(text.splitlines()):

            line = line.replace("\t", " & ")

            if 0.5 <= r < 0.75:
                if i%2 == 0:
                    line = "\\rowcolor{mycolor}\n" + line
            if 0.76 <= r < 1.0:
                if i%2 == 1:
                    line = "\\rowcolor{mycolor}\n" + line

            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _randomly_add_vert_lines(self, table_string, prob):
        # Split the string into lines
        lines = table_string.split('\n')

        # Find the line with the column definitions
        for i, line in enumerate(lines):
            if '{' in line and '}' in line:
                column_definitions = line
                break

        # Split the column definitions into individual columns
        columns = column_definitions.strip().strip('{}').split()

        # Randomly add pipes between the columns
        new_columns = []
        for column in columns:
            if column in ['l', 'c', 'r']:
                if random.random() < prob:  # % chance of adding a pipe
                    new_columns.append(column + ' |')
                else:
                    new_columns.append(column)
            else:
                new_columns.append(column)

        # Join the new columns back into a string
        new_column_definitions = '{ ' + ' '.join(new_columns) + ' }'

        # Replace the old column definitions with the new ones
        lines[i] = new_column_definitions

        # Join the lines back into a single string
        new_table_string = '\n'.join(lines)

        return new_table_string

    def convert_tsv_to_latex(self, 
                             text: str,
                             randomize_colors: bool = True,
                             randomize_vertical_lines: bool = True,
                             vertical_line_prob: int = 0.5
                             ) -> str:

        text = self.replace_special_to_latex(text)

        if randomize_colors:
            latex_text = self._randomize_colors(text)
        else:
            latex_text = self._return_latex_text(text)

        header1 = "\\usepackage{colortbl}\n\\usepackage{xcolor}\n" 
        colors = self._return_colors() #Simple return method to declutter
        print(colors)
        header2 = "\\begin{document}\n\\begin{table}\n"

        header = header1 + np.random.choice(colors) + header2

        # column_style = "\\begin{tabular}{| >{\\color{blue}}l c c |}\n\\hline\n"
        # column_style = "\\begin{tabular}{| l c c c c c c c |}\n\\hline\n"
        # column_style = "\\begin{tabular}{| l c c c |}\n\\hline\n"
        # column_style = "\\begin{tabular}{| l c c |}\n\\hline\n"

        column_style = "\\begin{tabular}{| l c c r c c r |}\n\\hline\n"
        
        if randomize_vertical_lines:
            column_style = self._randomly_add_vert_lines(column_style, vertical_line_prob)

        footer = "\\hline\n\\end{tabular}\n\\end{table}\n\\end{document}"

        formatted_latex_text = header + column_style + latex_text + footer
        
        return formatted_latex_text

    def _read_jsonl(self, file_path: str) -> dict:
        
        assert file_path.suffix == ".jsonl"
        data = []
        with open(file_path) as reader:
            for obj in reader:
                data.append(eval(obj))

        return data
    
    def _write_tex_file(self, tex_filepath, table_latex, prefix) -> None:
        margin = str(round(random.uniform(0.2, 0.8), 2))
        font_type = random.choice(font_types)
        width = random.choice(["0.3", "0.4", "0.5", "0.6"])

        prefix = prefix.replace("FONT_TYPE", font_type)
        prefix = prefix.replace("{MARGIN}", margin)
        prefix = prefix.replace("{RULE_WIDTH}", width)

        '''
        table_latex_lines = table_latex.split("\n")
        table_latex_lines_copy = table_latex_lines.copy()
        counter = 0
        for i, line in enumerate(table_latex_lines):
            if line == "\\hline" and counter == 0:

                table_latex_lines_copy.insert(i, "\\arrayrulecolor{black}\\arrayrulewidth={width}mm".replace("{width}", width))
                counter += 1
            elif line == "\\hline" and counter == 1:
                subwidth = str(round(float(width) - 0.1, 1))
                table_latex_lines_copy.insert(i, "\\arrayrulewidth={subwidth}mm".replace("{subwidth}", subwidth))
                counter += 1
            
            elif line == "\\hline" and i == len(table_latex_lines) - 2:
                table_latex_lines_copy.insert(i+2, "\\arrayrulewidth={width}mm".replace("{width}", width))
                counter += 1
        
        table_latex = "\n".join(table_latex_lines_copy)
        '''

        # latex_code = prefix + table_latex + "\n\end{document}"
        latex_code = prefix + table_latex
        with open(tex_filepath, 'w') as f:
            f.write(latex_code)

    def _crop_synth_table(self, image_path: str):
        # Load the image natrually and in grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply a binary threshold to the image
        _, binary = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest rectangular contour assuming it's the table
        table_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(table_contour)

        # Crop the table from the image
        height, width = gray_image.shape
        cropped_table = image[max(0, y-random.randint(10, 100)):min(y+h+random.randint(10, 100), height), 
                            max(0, x-random.randint(10, 100)):min(x+w+random.randint(10, 100), width)]
        
        keeptrying = True
        counter = 0
        while keeptrying:
            try:
                cv2.imwrite(image_path, cropped_table)
                keeptrying = False
            except Exception as e:
                print(e)
                time.sleep(0.5)
                counter += 1
                if counter >= 10:
                    break

        cv2.imwrite(image_path, cropped_table)

        # return cropped_table, cropped_table.shape
    
    def _image_augmentations(self, 
                             augmentations: List[str] = [
                                 "RandomBrightnessContrast",
                                 "Blur",
                                 "Downscale",
                                 "GaussNoise",
                                 "ISONoise",
                                 "ImageCompression",
                                 "RandomGamma",
                                 "ToGray"
                             ]
                             ):
        
        transformations_map = {
            "RandomBrightnessContrast": A.RandomBrightnessContrast(p=0.5),
            "Blur": A.Blur(p=0.5, blur_limit=(1,3)),
            "Downscale": A.Downscale(p=0.5, scale_min=0.55, scale_max=0.85),
            "GaussNoise": A.GaussNoise(p=0.5, var_limit=(50.0, 250.0)),
            "ISONoise": A.ISONoise(p=0.2, color_shift=(0.1, 0.5), intensity=(0.1, 0.5)),
            "ImageCompression": A.ImageCompression(p=0.5, quality_lower=50, quality_upper=90),
            "RandomGamma": A.RandomGamma(p=0.5, gamma_limit=(50, 150)),
            "ToGray": A.ToGray(p=0.5), 
        }

        assert all(item in transformations_map.keys() for item in augmentations), \
            f"Selected augmentations must be in: {transformations_map.keys()}."
        
        pipeline = []

        for name in augmentations:
            transformation = transformations_map.get(name)
            if transformation:
                pipeline.append(transformation)
        
        transform = A.Compose(pipeline)

        return transform
    
    def _generate_images(self, 
                         folder_name, 
                         data,
                         randomize_dpi: bool = True,
                         use_augmentations: bool = True) -> None:
    
        # for i, d in enumerate(data):
            # table_latex = d["latex"]
            # if "OVP PIN" in data:
            #     self._write_tex_file('table.tex', data, DOCUMENT_PREFIX)
            # else:
            #     self._write_tex_file('table.tex', data, DOCUMENT_PREFIX_LANDSCAPE)
        self._write_tex_file('table.tex', data, DOCUMENT_PREFIX_LANDSCAPE)

        # tex_filepath = Path(f"/{folder_name}/table_{i}.tex")
        tex_filepath = Path(f"/{folder_name}/table.tex")
        print("---TEX FILE PATH---")
        print(tex_filepath)
        pdf_filepath = tex_filepath.with_suffix(".pdf")
        print("---PDF FILE PATH---")
        print(pdf_filepath)
        img_filepath = tex_filepath.with_suffix(".jpg")

        # Compile the LaTeX file to PDF
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'])
        while not os.path.exists("./table.pdf"):
            time.sleep(0.5)
        subprocess.run(['mv', 'table.tex', tex_filepath])
        subprocess.run(['mv', 'table.pdf', pdf_filepath])

        #TODO: Randomize DPI
        if randomize_dpi:
            images = convert_from_path(pdf_filepath, dpi=np.random.choice([90, 120, 250, 300]))
        else:
            images = convert_from_path(pdf_filepath, dpi=250)
        image = images[-1] # Sometimes more than one page is generated, so we take the last one
        image_size = image.size  # This returns a tuple (width, height)
        print(image_size)

        if use_augmentations:
            transform = self._image_augmentations()
            image = transform(image=np.array(image))['image']
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

        image.save(img_filepath, 'JPEG')
        
        self._crop_synth_table(str(img_filepath))
        subprocess.run(['rm', tex_filepath])
        subprocess.run(['rm', pdf_filepath])
    
    def generate(self, data_dir) -> None:
        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".jsonl"): continue
            
            data_path = Path(os.path.join(data_dir, file_name))
            data = self._read_jsonl_data(data_path)
            subfolder_name = data_path.with_suffix('')
            subfolder_name = Path(str(subfolder_name))
            os.makedirs(subfolder_name, exist_ok=True)
            self._generate_images(subfolder_name, data)
    





import os
import cv2
import glob
import logging
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.core.composition import Compose
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

class TableTools:

    def convert_pdf_to_images(data_directory: str) -> None:
            """
            This method converts a pdf file to a series of images of each page
            
            Args:
                data_directory: Base path to save images.

            Raises:
                TypeError: If `data_directory` is not a string.
                FileNotFoundError: If `data_directory` does not exist.
            """

            assert isinstance(data_directory, str), \
                TypeError(f"Expected str, got {type(data_directory)}")

            # Get all pdf files in the directory and its 
            # subdirectories recursively
            try:
                files = glob.glob(data_directory + "**/**",
                                        recursive=True)
                files = [file for file in files if file.endswith(".pdf")]
            except FileNotFoundError:
                logging.error(f"{data_directory} not found")
                raise FileNotFoundError(f"{data_directory} does not exist.")

            for filename in files:
                logging.info(f"Converting {filename} to a folder of \
                             images in the same location")

                # Convert pdf to images and save them in the images 
                # subdirectory.
                images = convert_from_path(filename)
                base_folder = filename.split('.')[0]
                output_folder = os.path.join(base_folder, "images")
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
            data_directory: directory of images

        Raises:
            TypeError: If `data_directory` is not a string.
            TypeError: If `conf`, `iou`, `agnostic_nms`, `max_det`, \
                `threshold`, `offset` are not floats.
            TypeError: If `agnostic_nms`, `max_det` are not booleans.
            TypeError: If `max_det` is not an integer.
            FileNotFoundError: If `data_directory` does not exist.
        """

        assert isinstance(data_directory, str), \
            TypeError(f"Expected str, got {type(data_directory)}")
        assert isinstance(conf, float), \
            TypeError(f"Expected float, got {type(conf)}")
        assert isinstance(iou, float), \
            TypeError(f"Expected float, got {type(iou)}")
        assert isinstance(agnostic_nms, bool), \
            TypeError(f"Expected bool, got {type(agnostic_nms)}")
        assert isinstance(max_det, int), \
            TypeError(f"Expected int, got {type(max_det)}")
        assert isinstance(threshold, float), \
            TypeError(f"Expected float, got {type(threshold)}")
        assert isinstance(offset, int), \
            TypeError(f"Expected int, got {type(offset)}")
        
        # YOLO works quite well and has less issues than PaddleOCR tools.
        model = YOLO('foduucom/table-detection-and-extraction')

        # set model parameters
        model.overrides['conf'] = conf  # NMS confidence threshold
        model.overrides['iou'] = iou  # NMS IoU threshold
        model.overrides['agnostic_nms'] = agnostic_nms  # NMS class-agnostic
        model.overrides['max_det'] = max_det  # maximum number of detections per image

        # Get all files in directory recursively
        try:
            files = glob.glob(data_directory + "**/**", recursive=True)
            files = [file for file in files if file.endswith(".jpg")]
        except FileNotFoundError:
            logging.error(f"{data_directory} not found.")
            raise FileNotFoundError(f"{data_directory} does not exist.")

        # Iterate over all files and crop tables from them
        for filename in files:
            logging.info(f"Cropping tables from {filename}")
            results = model.predict(filename)
            
            # Get bounding boxes and mask for valid boxes with 
            # confidence >= threshold
            # boxes, mask = results[0].boxes.xyxy.numpy(), 
            boxes, = results[0].boxes.xyxy.numpy(),
            mask = results[0].boxes.conf.numpy() >= threshold
            
            # Get valid boxes and load image.
            valid_boxes = boxes[mask]
            loaded_img = Image.open(filename)
            output_folder = filename.partition('images')[0] + "cropped_tables/"

            # Create output folder if it doesn't exist.
            os.makedirs(output_folder, exist_ok=True)

            # Crop tables and save them to output folder.
            for i, box in enumerate(boxes):
                cropped_table = loaded_img.crop((box[0]-offset, 
                                                box[1]-offset, 
                                                box[2]+offset, 
                                                box[3]+offset))
                page_no = filename.partition('images')[-1].replace("/", "").replace(".jpg", "")
                output_path = f"{output_folder}/{page_no}_table_{i}.jpg"
                cropped_table.save(output_path, "JPEG")
        logging.info(f"Cropped tables saved to {data_directory} \
                     in subdirectories.")

    def replace_special_to_latex(self, text: str) -> str:
        
        """
        Replace special characters with their LaTeX equivalents.

        Args:
            text: Text to be converted.

        Returns:
            Converted text.

        Raises:
            TypeError: If `text` is not a string.
        """

        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        
        text = text.replace("%", "\\%")
        text = text.replace("_", "\\_")
        text = text.replace("*", "\\quad ")
        text = text.replace("$", "\\$")

        return text
    
    def _return_latex_text(self, text: str) -> None:

        """
        Return LaTeX formatted text from given text.

        Args:
            text: Text to be converted.

        Returns:
            Converted text.

        Raises:
            TypeError: If `text` is not a string.
        """

        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        
        latex_text = ""

        for line in text.splitlines():
            line = line.replace("\t", " & ")
            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _return_colors(self,
                       colors: List[tuple] = [
                           (150,210,255),
                           (210,210,210),
                           (175,250,180),
                           (250,250,180),
                       ]) -> List[str]:

        """
        Returns a list of RGB color codes.

        Args:
            colors: A list of RGB color codes as tuples.

        Returns:
            A list of RGB color codes in LaTeX format for table coloring.

        Raises:
            TypeError: If `colors` is not a list.
        """

        assert isinstance(colors, list), \
            TypeError(f"Expected List, got {type(colors)}.")

        colors = [f"\\definecolor{{mycolor}}{{RGB}}{{{','.join(map(str, color))}}}" \
                  for color in colors]
        
        return colors
    
    def _randomize_colors(self, text: str) -> str:

        """
        Randomly colorize rows of a table.

        Args:
            text: LaTeX text of a table.

        Returns:
            LaTeX text with randomly colored rows.
        """

        # Get a random number for color selection.
        r = np.random.uniform(0, 1)

        latex_text = ""

        for i ,line in enumerate(text.splitlines()):

            line = line.replace("\t", " & ")

            # If between 0.5 and 0.75, Colorize every other row
            # starting from second position.
            if 0.5 <= r < 0.75:
                if i%2 == 0:
                    line = "\\rowcolor{mycolor}\n" + line

            # If between 0.75 and 1.0, Colorize every other row
            # starting from first position
            if 0.76 <= r < 1.0:
                if i%2 == 1:
                    line = "\\rowcolor{mycolor}\n" + line

            line += " \\\\\n"
            latex_text += line

        return latex_text
    
    def _randomly_add_vert_lines(self, 
                                 table_string: str, 
                                 prob: float
                                 ) -> str:

        """
        Randomly add vertical lines between columens in a LaTeX table.

        Args:
            table_string: LaTeX string of a table.
            prob: Probability of adding a vertical line.

        Returns:
            LaTeX string with randomly added vertical lines.

        Raises:
            TypeError: If `table_string` is not a string.
            TypeError: If `prob` is not a float.
        """

        assert isinstance(table_string, str), \
            TypeError(f"Expected str, got {type(table_string)}.")
        assert isinstance(prob, float), \
            TypeError(f"Expected float, got {type(prob)}.")

        # Split the string into lines
        lines = table_string.split('\n')

        # Find the line with columns
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
                if random.random() < prob:  # chance of adding a pipe
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
                             column_style: str,
                             text: str,
                             randomize_colors: bool = True,
                             randomize_vertical_lines: bool = True,
                             vertical_line_prob: float = 0.5
                             ) -> str:
        
        """
        Convert a tsv string representation of a table to a LaTeX table.

        Args:
            column_style: LaTeX style for the columns.
            text: tsv string representation of a table.
            randomize_colors: If True, randomize the colors of the cells.
            randomize_vertical_lines: If True, randomize vertical lines between \
                columns.
            vertical_line_prob: Probability of adding a vertical line.

        Returns:
            LaTeX string representation of the table.

        Raises:
            TypeError: If `column_style` or `text` are not strings.
            TypeError: If `randomize_colors` or  \
                `randomize_vertical_lines` are not booleans.
            TypeError: If `vertical_line_prob` is not a float.
        """

        assert isinstance(column_style, str), \
            TypeError(f"Expected str, got {type(column_style)}.")
        assert isinstance(text, str), \
            TypeError(f"Expected str, got {type(text)}.")
        assert isinstance(randomize_colors, bool), \
            TypeError(f"Expected bool, got {type(randomize_colors)}.")
        assert isinstance(randomize_vertical_lines, bool), \
            TypeError(f"Expected bool, got {type(randomize_vertical_lines)}.")
        assert isinstance(vertical_line_prob, float), \
            TypeError(f"Expected float, got {type(vertical_line_prob)}.")

        # Replace special characters with their LaTeX equivalents
        text = self.replace_special_to_latex(text)

        # Do color randomization of rows if True
        if randomize_colors:
            logging.info("Randomizing row colors.")
            latex_text = self._randomize_colors(text)
        else:
            latex_text = self._return_latex_text(text)

        header1 = "\\usepackage{colortbl}\n\\usepackage{xcolor}\n" 
        colors = self._return_colors() #Simple return method to declutter
        header2 = "\\begin{document}\n\\begin{table}\n"

        header = header1 + np.random.choice(colors) + header2

        if randomize_vertical_lines:
            logging.info("Randomizing column lines.")
            column_style = self._randomly_add_vert_lines(column_style, vertical_line_prob)

        footer = "\\hline\n\\end{tabular}\n\\end{table}\n\\end{document}"

        # Create final LaTeX table.
        formatted_latex_text = header + column_style + latex_text + footer
        
        return formatted_latex_text

    def _read_jsonl(self, file_path: str) -> dict:

        """
        Reads a .jsonl file and returns a list of dictionaries.

        Args:
            file_path: The path to the .jsonl file.

        Returns:
            A list of dictionaries read from the .jsonl file.

        Raises:
            TypeError: If file_path is not a string.
            ValueError: If file_path does not have a .jsonl extension.
            FileNotFoundError: If the file does not exist.
        """
        
        assert isinstance(file_path, str), \
            TypeError(f"Expected str, got {type(file_path)}.")
        assert file_path.suffix == ".jsonl", \
            ValueError(f"Expected .jsonl file, got {file_path}.")
        
        data = []
        try:
            with open(file_path) as reader:
                for obj in reader:
                    data.append(eval(obj))
        except FileNotFoundError:
            logging.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")

        return data
    
    # TODO: Provide more control on font types.
    def _write_tex_file(self, tex_filepath: str, 
                        table_latex: str, 
                        prefix: str,
                        font_types: List[str] = [
                            "lmodern",
                            "times",
                            "helvet", 
                            "courier",
                            "mathpazo",
                            "newcent"
                        ],
                        ) -> None:
        
        """
        Writes a tex file as part of generating tables to images pipeline.

        Args:
            tex_filepath: The path to the tex file.
            table_latex: The LaTeX code for the table.
            prefix: The prefix for the tex file.
            font_types: The font types to use.

        Raises:
            TypeError: If `tex_filepath`, `table_latex`, \
                 or `prefix` are not strings.
            TypeError: If `font_types` is not a list.
            ValueError: If `font_types` contains an unsupported font type.
        """
        
        assert isinstance(tex_filepath, str), \
            TypeError(f"Expected str, got {type(tex_filepath)}.")
        assert isinstance(table_latex, str), \
            TypeError(f"Expected str, got {type(table_latex)}.")
        assert isinstance(prefix, str), \
            TypeError(f"Expected str, got {type(prefix)}.")
        assert isinstance(font_types, list), \
            TypeError(f"Expected list, got {type(font_type)}.")

        assert all(font_type for font_type in ["lmodern",
                                                "times",
                                                "helvet", 
                                                "courier",
                                                "mathpazo",
                                                "newcent"]), \
            ValueError(f"Invalid font type: {font_type}.")
        
        # Set formatting.
        margin = str(round(random.uniform(0.2, 0.8), 2))
        font_type = random.choice(font_types)
        width = random.choice(["0.3", "0.4", "0.5", "0.6"])

        # Replace placeholders in prefix.
        prefix = prefix.replace("FONT_TYPE", font_type)
        prefix = prefix.replace("{MARGIN}", margin)
        prefix = prefix.replace("{RULE_WIDTH}", width)
    
        latex_code = prefix + table_latex
        # Write to tex file.
        logging.info(f"Writing LaTeX table to {tex_filepath}.")
        with open(tex_filepath, 'w') as f:
            f.write(latex_code)

    def _crop_synth_table(self, image_path: str) -> None:

        """
        Crops the synthetic table image based on its border.

        Args:
            image_path: Path to the synthetic table image.
        
        Raises:
            TypeError: If image_path is not a string.
        """

        assert isinstance(image_path, str), \
            TypeError(f"Expected str, got {type(image_path)}")

        # Load the image natrually and in grayscale
        image = cv2.imread(image_path)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply a binary threshold to the grayscale image
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

        logging.info(f"Writing image to: {image_path}.")
        cv2.imwrite(image_path, cropped_table)
    
    # TODO: Provide control
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
                             ) -> Compose:
        
        """
        Args:
            augmentations: List of augmentations to apply.

        Returns:
            Composition of all the augmentations.

        Raises:
            TypeError: If augmentations is not a list.
            ValueError: If augmentations contains an unsupported augmentation.
        """

        assert isinstance(augmentations, list), \
            TypeError(f"Expected list, got {type(augmentations)}.")
        
        # TODO: allow for control over params.
        transformations_map = {
            "RandomBrightnessContrast": A.RandomBrightnessContrast(p=0.5),
            "Blur": A.Blur(p=0.5, blur_limit=(1,3)),
            "Downscale": A.Downscale(p=0.5, scale_min=0.75, scale_max=0.85),
            "GaussNoise": A.GaussNoise(p=0.5, var_limit=(50.0, 250.0)),
            "ISONoise": A.ISONoise(p=0.2, color_shift=(0.1, 0.5), intensity=(0.1, 0.5)),
            "ImageCompression": A.ImageCompression(p=0.5, quality_lower=65, quality_upper=90),
            "RandomGamma": A.RandomGamma(p=0.5, gamma_limit=(50, 150)),
            "ToGray": A.ToGray(p=0.5), 
        }

        assert all(item in transformations_map.keys() for item in augmentations), \
            ValueError(f"Selected augmentations must be in: {transformations_map.keys()}.")
        
        # Initialize the pipeline with the selected augmentations.
        pipeline = []

        # Add the selected augmentations to the pipeline.
        for name in augmentations:
            transformation = transformations_map.get(name)
            if transformation:
                pipeline.append(transformation)
        
        # Compose the augmentations into a single transform.
        transform = A.Compose(pipeline)

        return transform
    
    def generate_images(self, 
                         folder_name: str, 
                         data: str,
                         randomize_dpi: bool = True,
                         use_augmentations: bool = False) -> None:
        
        assert isinstance(folder_name, str), \
            TypeError(f"Expected str, got {type(folder_name)}.")
        assert isinstance(data, str), \
            TypeError(f"Expected str, got {type(data)}.")
        assert isinstance(randomize_dpi, bool), \
            TypeError(f"Expected bool, got {type(randomize_dpi)}.")
        assert isinstance(use_augmentations, bool), \
            TypeError(f"Expected bool, got {type(use_augmentations)}.")

        # Write intermediate tex file.
        self._write_tex_file('table.tex', data, DOCUMENT_PREFIX_LANDSCAPE)

        # Create new filepaths for file types.
        tex_filepath = Path(f"/{folder_name}/table.tex")
        pdf_filepath = tex_filepath.with_suffix(".pdf")
        img_filepath = tex_filepath.with_suffix(".jpg")

        # Compile the LaTeX file to PDF
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'table.tex'])
        while not os.path.exists("./table.pdf"):
            time.sleep(0.5)
        #Move files to respective paths.
        subprocess.run(['mv', 'table.tex', tex_filepath])
        subprocess.run(['mv', 'table.pdf', pdf_filepath])

        # Randomize DPI if True.
        if randomize_dpi:
            images = convert_from_path(pdf_filepath, dpi=np.random.choice([90, 120, 250]))
        else:
            images = convert_from_path(pdf_filepath, dpi=250)
        image = images[-1] # Sometimes more than one page is generated, so we take the last one

        # Do image augmentations if True.
        if use_augmentations:
            transform = self._image_augmentations()
            image = transform(image=np.array(image))['image']
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

        logging.info(f"Saving image to: {img_filepath}")
        image.save(img_filepath, 'JPEG')
        
        # Clean up intermediate files.
        self._crop_synth_table(str(img_filepath))
        subprocess.run(['rm', tex_filepath])
        subprocess.run(['rm', pdf_filepath])
    





import json
import os

import cv2
import PyPDF2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


# Function to combine two FPDF objects
def combine_fpdfs(buffer1, buffer2, output_filename):
    """combine_fpdfs.

    Parameters
    ----------
    buffer1 :
        buffer1
    buffer2 :
        buffer2
    output_filename :
        output_filename
    """
    # Create PDF reader objects from the bytes buffers
    buffer1.seek(0)
    buffer2.seek(0)
    pdf1 = PyPDF2.PdfReader(buffer1)
    pdf2 = PyPDF2.PdfReader(buffer2)

    # Create a PDF writer object for the output PDF
    pdf_writer = PyPDF2.PdfWriter()

    # Add pages from the first PDF
    for page_num in range(len(pdf1.pages)):
        page = pdf1.pages[page_num]
        # page = pdf1.getPage(page_num)
        pdf_writer.add_page(page)

    # Add pages from the second PDF
    for page_num in range(len(pdf2.pages)):
        page = pdf2.pages[page_num]
        pdf_writer.add_page(page)

    # Write the combined PDF to a file
    with open(output_filename, "wb") as output_file:
        pdf_writer.write(output_file)


def load_files(rootdir):
    """load_files.

    Parameters
    ----------
    rootdir :
        rootdir
    """
    file_dict = {}
    for rootdir, subdirs, filenames in os.walk(rootdir):
        for file_ in filenames:
            with open(os.path.join(rootdir, file_)) as f:
                file_dict[file_.replace(".txt", "")] = f.read()
    return file_dict


def load_prompts():
    """load_prompts."""
    return load_files(rootdir="prompts")


def load_constraints(experiment_type):
    """load_constraints.

    Parameters
    ----------
    experiment_type :
        experiment_type
    """
    constraints_dict = load_files(rootdir="constraints")
    constraints_dict = {
        key: constraints_dict[key][:-1].split("\n")
        for key in constraints_dict
        if experiment_type.lower() in key
    }
    return constraints_dict


def load_constraints_2():
    """load_constraints_2."""
    constraints_dict = {}
    for rootdir, subdir, filenames in os.walk("./constraints"):
        for file_ in filenames:
            if "hardware" in file_:
                with open(os.path.join(rootdir, file_)) as f:
                    constraints_dict["hardware"] = f.read().split("\n")
                    constraints_dict["hardware"] = [
                        h for h in constraints_dict["hardware"] if h != ""
                    ]
            if "reagents" in file_:
                with open(os.path.join(rootdir, file_)) as f:
                    constraints_dict["reagents"] = f.read().split("\n")
                    constraints_dict["reagents"] = [
                        r for r in constraints_dict["reagents"] if r != ""
                    ]
    return constraints_dict


# adapted from https://github.com/shoumikchow/bbox-visualizer
def draw_rectangle(
    img, bbox, bbox_color=(255, 255, 255), thickness=3, is_opaque=False, alpha=0.5
):
    """Draws the rectangle around the object

    Parameters
    ----------
    img : ndarray
        the actual image
    bbox : list
        a list containing x_min, y_min, x_max and y_max of the rectangle positions
    bbox_color : tuple, optional
        the color of the box, by default (255,255,255)
    thickness : int, optional
        thickness of the outline of the box, by default 3
    is_opaque : bool, optional
        if False, draws a solid rectangular outline. Else, a filled rectangle which is semi transparent, by default False
    alpha : float, optional
        strength of the opacity, by default 0.5

    Returns
    -------
    ndarray
        the image with the bounding box drawn
    """
    output = img.copy()
    if not is_opaque:
        cv2.rectangle(
            output, (bbox[0], bbox[1]), (bbox[2],
                                         bbox[3]), bbox_color, thickness
        )
    else:
        overlay = img.copy()

        cv2.rectangle(overlay, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), bbox_color, -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def add_T_label(
    img,
    label,
    bbox,
    size=1,
    thickness=2,
    draw_bg=True,
    text_bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
):
    """adds a T label to the rectangle, originating from the top of the rectangle

    Parameters
    ----------
    img : ndarray
        the image on which the T label is to be written/drawn, preferably the image with the rectangular bounding box drawn
    label : str
        the text (label) to be written
    bbox : list
        a list containing x_min, y_min, x_max and y_max of the rectangle positions
    size : int, optional
        size of the label, by default 1
    thickness : int, optional
        thickness of the label, by default 2
    draw_bg : bool, optional
        if True, draws the background of the text, else just the text is written, by default True
    text_bg_color : tuple, optional
        the background color of the label that is filled, by default (255, 255, 255)
    text_color : tuple, optional
        color of the text (label) to be written, by default (0, 0, 0)

    Returns
    -------
    ndarray
        the image with the T label drawn/written
    """

    (label_width, label_height), baseline = cv2.getTextSize(
        label, font, size, thickness
    )
    # draw vertical line
    x_center = (bbox[0] + bbox[2]) // 2
    y_top = bbox[1] - 50
    cv2.line(img, (x_center, bbox[1]), (x_center, y_top), text_bg_color, 3)

    # draw rectangle with label
    y_bottom = y_top
    y_top = y_bottom - label_height - 5
    x_left = x_center - (label_width // 2) - 5
    x_right = x_center + (label_width // 2) + 5
    if draw_bg:
        cv2.rectangle(img, (x_left, y_top - 30),
                      (x_right, y_bottom), text_bg_color, -1)
    cv2.putText(
        img,
        label,
        (x_left + 5, y_bottom - (8 * size)),
        font,
        size,
        text_color,
        thickness,
        cv2.LINE_AA,
    )

    return img


def load_knowledgebase(path):
    """load_knowledgebase.

    Parameters
    ----------
    path :
        path
    """
    with open(path) as f:
        knowledgebase = json.load(f)
    return knowledgebase
    
import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
import roboflow
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.detection.detection_base_model import NmsSetting
from autodistill.helpers import split_data
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLOE


@dataclass
class YOLOEBase(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, model: str = None):
        self.ontology = ontology
        self.classes = self.ontology.classes()
        if model:
            self.model = YOLOE(model)
        elif self.classes:
            # Text/Visual Prompt model
            # (YOLOE-11S, YOLOE-11M, YOLOE-11L,
            #  YOLOE-v8S, YOLOE-v8M, YOLOE-v8L)
            self.model = YOLOE("yoloe-11l-seg.pt")
            self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))
        else:
            # Prompt Free model
            # (YOLOE-11S-PF, YOLOE-11M-PF, YOLOE-11L-PF,
            #  YOLOE-v8S-PF, YOLOE-v8M-PF, YOLOE-v8L-PF)
            self.model = YOLOE("yoloe-11l-seg-pf.pt")

    def predict(self, input: str | np.ndarray | Image.Image) -> sv.Detections:
        model_results = self.model(input)[0]
        inference_results = sv.Detections.from_ultralytics(model_results)
        return inference_results

    def label(
        self,
        input_folder: str,
        extension: str = ".jpg",
        output_folder: str | None = None,
        human_in_the_loop: bool = False,
        roboflow_project: str | None = None,
        sahi: bool = False,
        record_confidence: bool = False,
        nms_settings: NmsSetting = NmsSetting.NONE,
    ) -> sv.DetectionDataset:
        """
        Label a dataset with the model.
        """
        if output_folder is None:
            output_folder = input_folder + "_labeled"

        os.makedirs(output_folder, exist_ok=True)

        image_paths = glob.glob(input_folder + "/*" + extension)
        detections_map = {}

        if sahi:
            slicer = sv.InferenceSlicer(callback=self.predict)

        progress_bar = tqdm(image_paths, desc="Labeling images")
        for f_path in progress_bar:
            progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

            image = cv2.imread(f_path)
            if sahi:
                detections = slicer(image)
            else:
                detections = self.predict(f_path)

            if nms_settings == NmsSetting.CLASS_SPECIFIC:
                detections = detections.with_nms()
            if nms_settings == NmsSetting.CLASS_AGNOSTIC:
                detections = detections.with_nms(class_agnostic=True)

            detections_map[f_path] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), image_paths, detections_map
        )

        dataset.as_yolo(
            output_folder + "/images",
            output_folder + "/annotations",
            min_image_area_percentage=0.01,
            data_yaml_path=output_folder + "/data.yaml",
        )

        if record_confidence:
            image_names = [os.path.basename(f_path) for f_path in image_paths]
            self._record_confidence_in_files(
                output_folder + "/annotations", image_names, detections_map
            )
        split_data(output_folder, record_confidence=record_confidence)

        if human_in_the_loop:
            roboflow.login()

            rf = roboflow.Roboflow()

            workspace = rf.workspace()

            workspace.upload_dataset(output_folder, project_name=roboflow_project)

        print("Labeled dataset created - ready for distillation.")
        return dataset


from roboflow import Roboflow
rf = Roboflow(api_key="mgDbHMdsllImLwB8BNVN")
project = rf.workspace("rf100-vl").project("mahjong-vtacs-mexax-m4vyu-sjtd")
version = project.version(2)
dataset = version.download("coco")
                
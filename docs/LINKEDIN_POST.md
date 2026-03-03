I trained a deep learning model to diagnose crop diseases from leaf photos, and then made it show its reasoning.

Plant diseases destroy up to 40% of global crops annually. The standard solution is to have a trained agronomist inspect the field in person, but for millions of smallholder farmers, that is not an option. A leaf starts showing spots, and the farmer is left guessing which disease it is and which treatment to apply. A wrong guess means wasted pesticide and continued crop damage.

So I built a system that takes a photograph of a leaf and identifies the disease in seconds, covering 15 conditions across pepper, potato, and tomato crops.

The model is EfficientNetB0 with transfer learning from ImageNet -- pretrained on millions of natural images, then adapted to recognize leaf pathology. Training happened in two phases: first I froze the backbone and trained just the classifier head, then I unfroze the deeper layers and fine-tuned with a 10x smaller learning rate to preserve what the model already knew about visual patterns.

The dataset has a 21:1 class imbalance (3,209 images of Tomato Yellow Leaf Curl Virus but only 152 of Potato Healthy), so I used inverse-frequency weighted loss to make sure the model did not ignore the rare classes.

The result: 99.6% accuracy and 0.996 macro F1 across all 15 classes on a held-out test set. An important caveat -- PlantVillage is a lab-controlled dataset with clean backgrounds and consistent lighting, so these numbers represent best-case performance. Published benchmarks routinely hit 95-99%+ on this dataset. The real challenge is field deployment where messy real-world conditions would bring accuracy down significantly.

The part I am most proud of is the explainability. Every prediction comes with a Grad-CAM heatmap that highlights exactly which regions of the leaf the model focused on. When the model says "Bacterial Spot," the heatmap lights up on the spots. When it says "Late Blight," it highlights the dark lesion patches. This is not just a nice visualization. It is what separates a useful tool from a black box that nobody trusts. An agronomist can look at the heatmap, verify the model is focusing on the right thing, and then act on the recommendation with confidence.

The system runs on an interactive Streamlit dashboard with 5 pages covering data exploration, model performance, a full Grad-CAM gallery, and live predictions where you upload a leaf photo and get a diagnosis with visual explanation and treatment recommendation.

Technical stack: Python, PyTorch, EfficientNetB0, Grad-CAM, OpenCV, Streamlit, Plotly.

Code and full documentation on GitHub. Link in comments.

#DeepLearning #ComputerVision #Agriculture #PyTorch #TransferLearning #GradCAM #PlantPathology #DataScience #MachineLearning #Python

from dataprep import DataPreparation
from train import ModelTrainer
from plot_results import PlotResults
import os

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[ Don't RUN ]<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# if __name__ == "__main__":

#     root_path = os.path.dirname(__file__)

#     base_path = os.path.join(root_path,'images/')
    
#     # Data Preparation
#     data_prep = DataPreparation(base_path)
#     data_prep.prepare_data()
    
#     # Model Training
#     trainer = ModelTrainer(base_path)
#     history = trainer.train()
    
#     # Plot Results
#     PlotResults.plot(history)
#     best_epoch = PlotResults.get_best_epoch(history)
    

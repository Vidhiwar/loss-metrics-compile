# loss-metrics-compile

RESEARCH-PAPER:  {

      TITLE: "Roughness Index and Roughness Distance for Benchmarking Medical Segmentation",

      CITE:  "https://arxiv.org/pdf/2103.12350",                                
              
      AUTHORS: ["Vidhiwar Singh Rathour",
                      "Kashu Yamakazi",
                           "Ngan Le"]}  
                                                
RESEARCH-PAPER:  {

      TITLE: "Invertible Residual Network with Regularization for Effective Medical Image Segmentation",

      CITE:  "https://arxiv.org/pdf/2103.09042",                                
              
      AUTHORS: ["Kashu Yamakazi",
                    "Vidhiwar Singh Rathour",
                            "Ngan Le"]}                                                       
                                                                                    
DIRECTORY-TREE: {

      res: "Directory: MSC Files",
      lib: {"Directory: Losses and Metrices":{
            boundaryLoss.py:  "Python: Pytorch implementation of Boundary Loss",
            roughnessIndex.py:  "Python: Pytorch implementation of Roughness Indexe"}}
            
      output: "Directory: Results are written",
      data: "Directory: contains the data",}
                                       
HOW-TO-USE: {

      Uno: "Make sure the required libraries (Torch, Nibabel, Tqdm, ... etc.,.) are installed",
      Dos: "Navigate to ./loss-metrics-compile",
      Tres: "Do: $ python <file.py>"}
                               

#  EOF
                    

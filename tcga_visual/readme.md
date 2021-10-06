# Transfer learning on TCGA-SKCM

## Overview
To better evaluate the method, the model is deployed to the TCGA skin cancer dataset. The model is directly applied to the TCGA dataset without fine-tuning since patch-level labels are not available for TCGA dataset. For some TCGA slides, there are some annotations on them, but we are not sure what the annotations are made for. Even though the color, scan method and many other factors make the TCGA dataset very different from the melanocytic dataset used in the paper, the model is able to find out annotated regions in those TCGA slides. See some examples below.

<img src="./images/TCGA-D3-A2JD-06Z-00-DX1_B6DBA83D-6C77-4F73-87B8-30487C8AB7C1.png" width="400"> 
<img src="./images/TCGA-D3-A2JD-06Z-00-DX1_B6DBA83D-6C77-4F73-87B8-30487C8AB7C1_heat_.png" width="400">


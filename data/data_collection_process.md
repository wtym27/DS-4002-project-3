## How to download data
1. On the HPC terminal, go to the data/ folder with `cd data/`
2. In a browser, go to https://lila.science/datasets/nacti
3. Copy one of the links for the Images(1/4) folder
4. Back in the terminal, download the data file with `wget [PASTE LINK HERE]`
5. Wait
6. Unzip the file with `unzip [FILE NAME]`
7. Wait
8. Repeat steps 2-7 with Images(2/4), (3/4), (4/4), and Metadata (.csv).

The final data/ directory should look like this: 
```
DS-4002-project-3/
│── data/
│   ├── nacti_metadata.csv
│   ├── part0/
│   ├── part1/
│   ├── part2/
│   └── part3/
```
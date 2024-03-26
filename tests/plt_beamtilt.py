import numpy as np
import matplotlib.pyplot as plt

data = np.load("/scratch/erice/elferich/processing_thp1_ribosomes_unbinned/Assets/Parameters/refine_ctf_input_star_7_results.npy",allow_pickle=True)[0]


datasets = [20230317,20230505,20230628]

pltdata = {}

for dat in datasets:
    pltdata[dat] = {}
    pltdata[dat]["beam_tilt_x"] = []
    pltdata[dat]["beam_tilt_y"] = []
    pltdata[dat]["bis_x"] = []
    pltdata[dat]["bis_y"] = []
    pltdata[dat]["is_x"] = []
    pltdata[dat]["is_y"] = []



for key, value in data.items():
    _, dataset, bisx, bisy = key.split("_")
    dataset = int(dataset)
    bltx = value["beam_tilt_x"]
    blty = value["beam_tilt_y"]
    isx = value["particle_shift_x"]
    isy = value["particle_shift_y"]
    pltdata[dataset]["beam_tilt_x"].append(bltx)
    pltdata[dataset]["beam_tilt_y"].append(blty)
    pltdata[dataset]["is_x"].append(isx)
    pltdata[dataset]["is_y"].append(isy)
    pltdata[dataset]["bis_x"].append(float(bisx))
    pltdata[dataset]["bis_y"].append(float(bisy))
   

fig, axs = plt.subplots(3,2)
for i, dataset in enumerate(datasets):
    sc = axs[i][0].scatter(pltdata[dataset]["bis_x"],pltdata[dataset]["bis_y"],c=pltdata[dataset]["beam_tilt_x"],vmin=-0.002,vmax=0.0005)
    sc = axs[i][1].scatter(pltdata[dataset]["bis_x"],pltdata[dataset]["bis_y"],c=pltdata[dataset]["beam_tilt_y"],vmin=-0.002,vmax=0.0005)
    axs[i][0].set_title(f"Dataset {dataset} Beamtilt X")
    axs[i][1].set_title(f"Dataset {dataset} Beamtilt Y")
    axs[i][0].set_xlim(-6,6)
    axs[i][0].set_ylim(-6,6)
    axs[i][1].set_xlim(-6,6)
    axs[i][1].set_ylim(-6,6)
    # Show colorbar
    cbar = fig.colorbar(sc)
plt.show()

fig, axs = plt.subplots(1,2)
for i, dataset in enumerate(datasets):
    axs[0].plot(pltdata[dataset]["beam_tilt_x"],pltdata[dataset]["is_x"],'o')
    axs[1].plot(pltdata[dataset]["beam_tilt_y"],pltdata[dataset]["is_y"],'o')
axs[0].set_xlabel("Beamtilt X")
axs[0].set_ylabel("Particle Shift X")
axs[1].set_xlabel("Beamtilt Y")
axs[1].set_ylabel("Particle Shift Y")
plt.show()
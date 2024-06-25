import json
from tkinter.filedialog import asksaveasfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from tkinter import ttk
import tkinter


def change_theme(style, root, toolbar):
    if style.theme_use() == 'alt':
        style.theme_use('clam')
        style.map('TButton', foreground=[('!active', 'green'), ('pressed', 'red'), ('active', 'white')],
                  background=[('!active', '#191A20'), ('pressed', 'green'), ('active', 'black')]
                  )
        style.configure('TFrame', foreground='green', background='#191A20')
        style.map('TMenubutton', foreground=[('!active', 'green'), ('pressed', 'red'), ('active', 'white')],
                  background=[('!active', '#191A20'), ('pressed', 'green'), ('active', 'black')])
        style.configure('Entry', foreground='green', background='black')
        root.configure(background='#191A20')
        toolbar.config(background="#191A20")
        toolbar._message_label.config(background="#191A20", foreground="green")
        toolbar.update()
        style.configure('TLabel', foreground='green', background='#191A20')
        style.configure('TCheckbutton',foreground='green', background='#191A20')
    else:
        style.theme_use('alt')
        style.configure('TFrame', foreground='green', background='#B7B7B7')
        root.configure(background='#B7B7B7')
        toolbar.config(background="#B7B7B7")
        toolbar._message_label.config(background="#B7B7B7", foreground="white")
        toolbar.update()
        style.configure('TLabel', foreground='black', background='#B7B7B7')


class PrintLogger():  # create file like object
    def __init__(self, textbox):  # pass reference to text widget

        self.textbox = textbox  # keep ref
        self.textbox.config(state=DISABLED)

    def write(self, text):
        self.textbox.config(state=NORMAL)
        self.textbox.insert(END, text)  # write text to textbox
        # could also scroll to end of textbox here to make sure always visible
        self.textbox.config(state=DISABLED)

    def flush(self):  # needed for file like object
        pass


def calcDistances(inputData, outputData):
    print("itt vagyunk")
    from sklearn.metrics.pairwise import euclidean_distances
    from numpy import genfromtxt
    """
    with open(input, 'r', encoding='utf-8-sig') as f:
        inputData = np.transpose(genfromtxt(f, dtype=int, delimiter=';'))
    with open(output, 'r', encoding='utf-8-sig') as f:
        outputData = np.transpose(genfromtxt(f, dtype=int, delimiter=';'))
   
    abd_distances = euclidean_distances(inputData, inputData)
    com_distances = euclidean_distances(outputData, outputData)


    abd_distances = (abd_distances - np.min(abd_distances)) / np.ptp(abd_distances)

    com_distances = (com_distances - np.min(com_distances)) / np.ptp(com_distances)
    distances = abs(abd_distances - com_distances)

    # distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    result = np.where(distances == np.amax(distances))
    print("maximumhelyek")
    print(result)
    print(distances[result[0][0], result[0][1]])

    abd_distances[result[0][0]][result[0][0]] = 100
    result2 = np.where(abd_distances == np.amin(abd_distances[result[0][0]]))
    print(result2)
    # print(result[0][0])

    # np.savetxt"distances.csv", distances, delimiter=';')
     """
    distances = euclidean_distances(inputData, outputData)
    diagonal=distances.diagonal
    return [distances,diagonal]


def heatmap(input, output):
    print(np.load(input).shape)
    input = np.load(input).squeeze()#[:,3000:6000]
    print(input.shape)
    output = np.load(output).squeeze()#[:,3000:6000]

    [distances,diagonal] = calcDistances(input, output)#[0]
    print(diagonal)
    figure1.clear()
    ax1 = figure1.add_subplot(111)

    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    import matplotlib.pyplot as plt
    import math

    # fig, (ax1, ax2,ax3) = plt.subplots(1, 2)
    afont = {'fontname': 'Arial'}
    # fig.suptitle('Principal Component Analyses', **afont)

    img = ax1.imshow(distances, cmap="rainbow")
    plt.colorbar(img, ax=ax1, orientation="horizontal", label='Normalized distances')
    # ax1.colorbar(orientation="horizontal", label='Normalized distances')  # , shrink=.5)

    ax1.set_xlabel("Brain Regions", **afont)
    ax1.set_ylabel("Brain regions", **afont)
    ax1.set_title("A", **afont)  # Distances of the distances between brain regions' inputs and outputs", pad=8)

    from PIL import Image
    from io import BytesIO
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    figure1.savefig(png1, format='png')

    # (2) load this image into PIL
    png2 = Image.open(png1)
    (width, height) = png2.size
    png3 = png2.resize((int(math.floor(width * 1.5)), int(math.floor(height * 1.5))), Image.ANTIALIAS)
    # (3) save as TIFF
    # png3.save(Figures/Fig4.tiff', dpi=(600, 600))
    png1.close()
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()


def createColors(region_names):
    ##Generate color for each Region Type##
    region_types = []
    for region in region_names:
        print(region.split("\\"))
        region_types.append(region.split("\\")[3].split("_")[3][:-4])
    print(region_types)

    region_types_set = sorted(list(set(region_types)))
    print(region_types_set)

    from random import randint
    colors = []
    n = len(region_types_set)
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return [colors, region_types_set, region_names, region_types]


def calcPCA2(input, counter):
    from numpy import genfromtxt
    import numpy as np
    inputData = np.squeeze(input)

    ##Do the PCA##
    from sklearn.decomposition import PCA

    # X = np.transpose(np.array(np.delete(inputData, 0, 1)))
    X = np.transpose(inputData)
    print(X)

    pca = PCA(n_components=2)
    pca.fit(X)

    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)
    print("PCA components:")
    print(np.argmax(pca.components_, axis=1))
    print("variance: ", pca.explained_variance_ratio_)
    eigenvectors = pca.components_
    print("length of eigenvectors: ",len(eigenvectors))
    eigenvectors[0][np.argmax(pca.components_, axis=1)[0]] = 0
    eigenvectors[1][np.argmax(pca.components_, axis=1)[1]] = 0
    print(np.argmax(eigenvectors, axis=1))
    from tkinter.filedialog import asksaveasfile

    """
    pca2 = PCA(n_components=524)
    pca2.fit(X)
    eigenvectors = pca2.components_
    print(len(eigenvectors))
    feature_importance=np.zeros(len(eigenvectors[0]))
    for i in range(len(eigenvectors)):
        feature_importance+=pca2.explained_variance_ratio_[i] * 1 / sum(abs(eigenvectors[i])) * abs(eigenvectors[i])
    print("Feature Importance")
    print(feature_importance)
    
    with open(asksaveasfile(initialfile='feature_importance.npy',
                            defaultextension=".npy").name, 'wb') as f:
        np.save(f, feature_importance)

    print(sum(feature_importance))
    """
    with open(asksaveasfile(initialfile='PCA.csv',
                            defaultextension=".csv").name, 'wb') as f:
        np.savetxt(f, X_pca, delimiter=";")

    with open(asksaveasfile(initialfile = 'PCA.npy',
defaultextension=".npy").name, 'wb') as f:
        np.save(f, X_pca)

    # X_new = pca.inverse_transform(X_pca)
    # plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    return X_pca


def pca(input, output):
    """
    import glob
    files = sorted(glob.glob("D:\phd\CytoCastNew/files2upload/abd*.tsv"))
    colorsandtypes = createColors(region_names=files)
    colors = colorsandtypes[0]
    colorsdata={}
    colorsdata["colors"]=colorsandtypes
    with open("D:\phd\CytoCastNew/colors.json", 'w') as outfile:
        outfile.write(json.dumps(colorsdata, indent=4, sort_keys=True))
    """
    figure1.clear()
    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    colorsandtypes = colorsdata["colors"]
    colors = colorsandtypes[0]
    region_types_set = colorsandtypes[1]
    region_names = colorsandtypes[2]
    region_types = colorsandtypes[3]
    import math
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 2)
    afont = {'fontname': 'Arial'}
    # fig.suptitle('Principal Component Analyses',**afont)
    ax1 = figure1.add_subplot(221)
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    ax1.set_title("PCA")
    ax2 = figure1.add_subplot(222)
    ax3 = figure1.add_subplot(223)
    # ax1.plot(x, y)
    # ax2.plot(x, -y)
    import numpy as np
    input = np.load(input)
    #input=np.loadtxt(input,
    #          delimiter=";", dtype=int).T
    #input = calcPCA2(input, "input")
    #print(input.shape)
    output = np.load(output)
    #output = np.loadtxt(input,
    #                   delimiter=";", dtype=int).T
    #merged=np.vstack((input,output)).T
    print("rang: ",np.linalg.matrix_rank(input))
    output = calcPCA2(input.T, "output")

    classes=""
    for i in range(len(colorsandtypes[1])):
        classes+=colorsandtypes[1][i]+"={mark=square*,"+colorsandtypes[0][i]+"},"
    print(classes)
    with open(asksaveasfile(initialfile='PCA_colored.csv',
                            defaultextension=".csv").name, 'w') as f:

        print('This message will be written to a file.')
        for i in range(len(input)):#(len(region_names)):
            print(str(input[i, 0])+" "+ str(input[i, 1])+" "+region_types_set[region_types_set.index(region_types[i])]+" "+str(output[i, 0])+" "+ str(output[i, 1])+" "+region_types_set[region_types_set.index(region_types[i])],file=f)
            ax1.scatter(input[i, 0], input[i, 1], region_types_set.index(region_types[i]),
                        color=colors[region_types_set.index(region_types[i])])  # input[i, 1]
            ax2.scatter(output[i, 0], output[i, 1], region_types_set.index(region_types[i]),
                        color=colors[region_types_set.index(region_types[i])])  # output[i, 1]

    import matplotlib.patches as mpatches
    pops = []
    for i in range(len(colors)):
        # legend
        pops.append(mpatches.Patch(color=colors[i], label=region_types_set[i]))
    # plt.axis('equal');
    # Shrink current axis's height by 10% on the bottom
    """
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])           
    """

    ax1.set_xlabel("Principal Component 1", **afont)
    ax1.set_ylabel("Principal Component 2", **afont)
    ax2.set_xlabel("Principal Component 1", **afont)
    ax2.set_ylabel("Principal Component 2", **afont)
    ax1.set_title("A", **afont)
    ax2.set_title("B", **afont)

    # fig.subplots_adjust(bottom=0, wspace=0.33)
    ax3.axis("off")
    import matplotlib.font_manager as font_manager

    font = font_manager.FontProperties(family='Arial',
                                       weight='normal',
                                       style='normal', size=12)
    figure1.legend(handles=pops, bbox_to_anchor=(0.5, 0.3),
                   loc='center', ncol=5, numpoints=1, borderaxespad=0.1, prop=font)  # facecolor="plum"
    figure1.tight_layout()
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()
    from PIL import Image
    from io import BytesIO
    # (1) save the image in memory in PNG format
    png1 = BytesIO()
    figure1.savefig(png1, format='png')

    # (2) load this image into PIL
    png2 = Image.open(png1)
    (width, height) = png2.size
    pngresized = png2.resize((int(math.floor(width * 1.5)), int(math.floor(height * 1.5))), Image.ANTIALIAS)
    (width, height) = pngresized.size

    # (3) save as TIFF
    pngresized.save('D:\phd\CytoCastNew/Figures/Fig5.tiff', dpi=(600, 600))
    png1.close()


def kMeans(X, n_clusters: int, random_state: int) -> None:
    # Instantiate the KMeans models
    #
    from sklearn import datasets
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    #
    # Fit the KMeans model
    #
    km.fit_predict(X)
    #
    # Calculate Silhoutte Score
    #
    score = silhouette_score(X, km.labels_, metric='euclidean')
    #
    # Print the score
    #
    print('Silhouetter Score: %.3f' % score)

    from yellowbrick.cluster import SilhouetteVisualizer
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 8))
    ax2.flatten()
    for i in [2, 3, 4, 5]:
        '''
        Create KMeans instance for different number of clusters
        https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
        visualizer.ax.set_xlabel("silhouette score")
        visualizer.ax.set_ylabel("simulations")
        visualizer.fit(X)
        fig.show()

        # Getting the Centroids
        centroids = km.cluster_centers_
        label = km.fit_predict(X)

        u_labels = np.unique(label)

        # plotting the results:
        if(i==2):
            k=0
            l=0
        if (i == 3):
            k = 0
            l = 1
        if (i == 4):
            k = 1
            l = 0
        if (i == 5):
            k = 1
            l = 1
        for j in u_labels:
            ax2[k,l].scatter(X[label == j, 0], X[label == j, 1], label=j)
        ax2[k,l].scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    fig2.legend()
    fig2.show()


def plotAbs(WT,Hom, regions, elseboundary, isBar):
    exp = Experiment(fd.askopenfilenames(title='Experiment Binaries'))
    print(isBar)
    prots={'Q05586':	'NMDAR',
'P42261':	'AMPAR',
'P78352':	'PSD-95',
'Q96PV0':	'SYNGAP',
'Q5VSY0':	'GKAP',
'Q9Y566':	'SHank1',
'Q86YM7':	'Homer1',
'Q9Y566M':	'Shank1M'
}

    figure1.clear()
    import matplotlib.pyplot as plt
    from numpy import genfromtxt
    print(elseboundary)
    WTData = np.load(WT)
    print(WTData.shape)

    HomData=np.load(Hom)

    #HetData = np.load(Het)
    afont = {'fontname': 'Arial'}

    labels = np.arange(7)
    # sizes = [15, 30, 45, 10]
    colors = {
        0: "blue",
        1: "orange",
        2: "green",
        3: "red",
        4: "purple",
        5: "brown",
        6: "pink"
    }

    from random import randint
    colors2 = {}
    print( WTData.shape[1])
    for i in range(0, WTData.shape[1]):
        # colors.append()
        colors2[str(i)] = '#%06X' % randint(0, 0xFFFFFF)
    colors2["other"] = "pink"

    with open("D:\phd\CytoCastNew/output/colors2.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(colors2, f, indent=2)
    """
    with open("D:\phd\CytoCastNew/colors2.json", 'r') as f:
        colors2 = json.load(f)
    """

    fontsize=10
    if(isBar):
        fontsize=5
    """
    with open(folder_selected_temp + 'colors.json') as f:
        colors2 = json.load(f)
    f.close()
    """
    ax1 = figure1.add_subplot(221)
    abper = []
    abnev = []
    egyeb = 0
    count = 0
    print(len(WTData[regions[0]]))
    if(isBar):
        stds_WT = np.load(fd.askopenfilename(title="STDS WT"))
        stds_Hom = np.load(fd.askopenfilename(title="STDS MUT"))
        #stds_Het = np.load(fd.askopenfilename())
        errors=[]
    for item in WTData[regions[0]]:

        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            print(count)
            if(isBar):
                errors.append(stds_WT[regions[0]][count])
            temp2=[]
            temp1=exp.complexes.complexes[count].structure
            for titem in temp1:
                temp2.append(prots[titem])
            print(temp2)
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)
    explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    #patches, texts = ax1.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
    #                         shadow=False, startangle=90, rotatelabels=180, textprops={'fontsize': fontsize})
    # plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.35, .5), fontsize=8)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    tableax1="complex\tabundance\tstd\n"

    if(isBar):
        colorbar=[]
        for i in range(len(labels)):
            tableax1+=str(labels[i])+"\t"+str(sizes[i])+"\t"+str(errors[i])+"\n"
        print(tableax1)
        for label in labels:
            colorbar.append(colors2[label])

        ax1.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black',color=colorbar)
        ax1.set_xticks(labels)
        ax1.set_xticklabels(labels,rotation = 'vertical')
    else:
        patches, texts = ax1.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                          shadow=False, startangle=90, rotatelabels=180, textprops={'fontsize': fontsize})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


    ax1.set_title("A")


    ax2 = figure1.add_subplot(222)
    abper = []
    abnev = []
    egyeb = 0
    count = 0
    if(isBar):
        errors=[]
    for item in HomData[regions[0]]:
        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            if (isBar):
                errors.append(stds_Hom[regions[0]][count])
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)
    tableax2 = "complex\tabundance\tstd\n"
    if (isBar):
        colorbar = []
        for i in range(len(labels)):
            tableax2+=str(labels[i])+"\t"+str(sizes[i])+"\t"+str(errors[i])+"\n"
        print(tableax2)
        for label in labels:
            colorbar.append(colors2[label])

        ax2.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black', color=colorbar)
        ax2.set_xticks(labels)
        ax2.set_xticklabels(labels, rotation='vertical')
    else:
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        patches, texts = ax2.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                                 shadow=False, startangle=90, rotatelabels=180, textprops={'fontsize': fontsize})
        # plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.35, .5), fontsize=8)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("B")
    """
    ax3 = figure1.add_subplot(233)
    abper = []
    abnev = []
    egyeb = 0
    count = 0
    if(isBar):
        errors=[]
    for item in HetData[regions[0]]:
        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            if (isBar):
                errors.append(stds_Het[regions[0]][count])
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)
    if (isBar):
        colorbar = []
        for label in labels:
            colorbar.append(colors2[label])

        ax3.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black', color=colorbar)
        ax3.set_xticks(labels)
        ax3.set_xticklabels(labels, rotation='vertical')
    else:
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        patches, texts = ax3.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                                 shadow=False, startangle=90, rotatelabels=180, textprops={'fontsize': fontsize})
        # plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.35, .5), fontsize=8)
        ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax3.set_title("C")
    """
    ax4 = figure1.add_subplot(223)

    abper = []
    abnev = []
    egyeb = 0
    count = 0
    if(isBar):
        errors=[]
    for item in WTData[regions[1]]:
        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            if (isBar):
                errors.append(stds_WT[regions[1]][count])
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)

    tableax4 = "complex\tabundance\tstd\n"
    if (isBar):
        colorbar = []
        for i in range(len(labels)):
            tableax4+=str(labels[i])+"\t"+str(sizes[i])+"\t"+str(errors[i])+"\n"
        print(tableax4)
        for label in labels:
            colorbar.append(colors2[label])

        ax4.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black', color=colorbar)
        ax4.set_xticks(labels)
        ax4.set_xticklabels(labels, rotation='vertical')
    else:
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        patches, texts = ax4.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                                 shadow=False, startangle=90, rotatelabels=180,textprops={'fontsize': fontsize})
        # plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.35, .5), fontsize=8)
        #plt.legend(patches, labels, loc='lower left', fontsize=8, ncol=20) # bbox_to_anchor=(-0.35, .5),
        ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax4.set_title("D")
    ax5 = figure1.add_subplot(224)

    abper = []
    abnev = []
    egyeb = 0
    count = 0
    if(isBar):
        errors=[]
    for item in HomData[regions[1]]:
        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            if (isBar):
                errors.append(stds_Hom[regions[1]][count])
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)
    tableax5 = "complex\tabundance\tstd\n"
    if (isBar):
        colorbar = []
        for i in range(len(labels)):
            tableax5+=str(labels[i])+"\t"+str(sizes[i])+"\t"+str(errors[i])+"\n"
        print(tableax5)
        for label in labels:
            colorbar.append(colors2[label])

        ax5.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black', color=colorbar)
        ax5.set_xticks(labels)
        ax5.set_xticklabels(labels, rotation='vertical')
    else:
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        ax5.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                shadow=False, startangle=90, rotatelabels=180,textprops={'fontsize': fontsize})
        ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax5.set_title("E")
    """
    ax6 = figure1.add_subplot(236)

    abper = []
    abnev = []
    egyeb = 0
    count = 0
    if(isBar):
        errors=[]
    for item in HetData[regions[1]]:
        if item < elseboundary:
            egyeb += item
        else:
            abper.append(item)
            abnev.append(str(count))
            if (isBar):
                errors.append(stds_Het[regions[1]][count])
        count += 1

    abper.append(egyeb)
    abnev.append("other")
    if(isBar):
        errors.append(0)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = abnev
    sizes = abper
    print(sizes)

    if (isBar):
        colorbar = []
        for label in labels:
            colorbar.append(colors2[label])

        ax6.bar(labels, sizes, yerr=errors, align='center', alpha=1, ecolor='black', color=colorbar)
        ax6.set_xticks(labels)
        ax6.set_xticklabels(labels, rotation='vertical')
    else:
        explode = (0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        ax6.pie(sizes, labels=labels, colors=[colors2[key] for key in labels],  # autopct='%1.1f%%',
                shadow=False, startangle=90, rotatelabels=180,textprops={'fontsize': fontsize})
        ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax6.set_title("F")
    """
    if(isBar):
        figure1.canvas.figure.subplots_adjust(wspace=0, hspace=0)
    figure1.tight_layout()
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()

def featureImportance():
    feature_importance=np.load(fd.askopenfilename())
    #print(feature_importance)
    # Legnagyobb 10 elem indexei
    top_10_indices = np.argsort(feature_importance)[-10:]

    # Fordítsd meg az indexek sorrendjét, hogy csökkenő sorrendben legyenek
    top_10_indices = top_10_indices[::-1]

    # Legnagyobb 10 elem
    top_10_values = feature_importance[top_10_indices]

    print("Legnagyobb 10 elem indexei:", top_10_indices)
    print("Legnagyobb 10 elem értékei:", top_10_values)
    #print(np.argmax(feature_importance))
    #print(feature_importance[np.argmax(feature_importance)])
    #print(sum(feature_importance))
    #print(np.mean(feature_importance))
    #print(np.median(feature_importance))
    #print(np.std(feature_importance))


def complex(i):
    prots = {'Q05586': 'NMDAR',
             'P42261': 'AMPAR',
             'P78352': 'PSD-95',
             'Q96PV0': 'SYNGAP',
             'Q5VSY0': 'GKAP',
             'Q9Y566': 'SHank1',
             'Q86YM7': 'Homer1',
             'Q9Y566M': 'Shank1M'
             }
    from google.protobuf.json_format import MessageToDict
    exp = Experiment(fd.askopenfilenames(title="Experiment Binaries"))
    print([field.name for field in exp.complexes.DESCRIPTOR.fields])
    print([field.name for field in exp.complexes.complexes.__getitem__(0).DESCRIPTOR.fields])
    structure=[]
    for item in exp.complexes.complexes.__getitem__(i).structure:
        structure.append(prots[item])
    print(structure)

def complexSize():
    exp = Experiment(fd.askopenfilenames(title="Experiment Binaries"))
    print([field.name for field in exp.complexes.DESCRIPTOR.fields])
    print([field.name for field in exp.complexes.complexes.__getitem__(0).DESCRIPTOR.fields])
    complexsizes = []
    for item in exp.complexes.complexes:
        complexsizes.append(len(item.structure))

    abundancesWT=np.load(fd.askopenfilename(title="Wild-type abundances"))
    abundancesMUT = np.load(fd.askopenfilename(title="Mutant abundances"))
    complexsizesAllWT=[]
    for i in range(524):
        dummysizeabundances = np.zeros(np.max(complexsizes) + 1)
        for j in range(len(abundancesWT[i])):
            dummysizeabundances[complexsizes[j]]+=abundancesWT[i][j]
        complexsizesAllWT.append(dummysizeabundances)
    complexsizesAllMUT = []
    with open(fd.asksaveasfilename(title="Save WT size abds"), 'wb') as f:
        np.save(f,complexsizesAllWT)

    for i in range(524):
        dummysizeabundances = np.zeros(np.max(complexsizes) + 1)
        for j in range(len(abundancesMUT[i])):
            dummysizeabundances[complexsizes[j]] += abundancesMUT[i][j]
        complexsizesAllMUT.append(dummysizeabundances)
    with open(fd.asksaveasfilename(title="Save MUT size abds"), 'wb') as f:
        np.save(f,complexsizesAllMUT)

def plotComplexSize(i):
    abundancesWT = np.load(fd.askopenfilename(title="Wild-type size abundances"))
    abundancesMUT = np.load(fd.askopenfilename(title="Mutant size abundances"))

    figure1.clear()
    ax1=figure1.add_subplot(111)


    # red dashes, blue squares and green triangles
    ax1.plot(abundancesWT[i], 'bs')
    ax1.plot(abundancesMUT[i], 'r')
    ax1.set_xlabel('Complex size')
    ax1.set_ylabel('Abundance')
    ax1.legend('wild-type','mutant')
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()



class Experiment:
    def __init__(self, filename) -> object:
        from multiPPComplexes_pb2 import MultiPPComplexes
        self.complexes = MultiPPComplexes()
        for file in filename:
            with open(file, "rb") as infile:
                self.complexes.MergeFromString(infile.read())


from tkinter import *
from tkinter import filedialog as fd


def load():
    from google.protobuf.json_format import MessageToDict
    exp = Experiment(fd.askopenfilenames())
    print([field.name for field in exp.complexes.DESCRIPTOR.fields])
    print([field.name for field in exp.complexes.complexes.__getitem__(0).DESCRIPTOR.fields])
    IDs = []
    i: int
    for i in range(len(exp.complexes.complexes)):
        IDs.append(exp.complexes.complexes.__getitem__(i).ID)
    print(len(IDs))
    print(sorted(IDs)[-1])

    # print(exp.complexes.complexes.__getitem__(10).ID)
    print(MessageToDict(exp.complexes.complexes.__getitem__(10).occurrences[0]))
    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    colorsandtypes = colorsdata["colors"]
    colors = colorsandtypes[0]
    region_types_set = colorsandtypes[1]
    region_names = colorsandtypes[2]
    region_types = colorsandtypes[3]
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        for i in range(len(IDs)):
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == region.split("\\")[3][:-4].replace('_',
                                                                        '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    abds_of_one_condition[i] = np.mean(abd.values)
                    print("siker")
        abds_of_all.append(abds_of_one_condition)
    with open('D:\\phd\\CytoCastNew/output/stds-wild-type.npy', 'wb') as f:
        np.save(f, abds_of_all)
    print(np.array(abds_of_all).shape)
    """
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        for i in range(len(IDs)):
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == 'M-Het-' + region.split("\\")[3][:-4].replace('_',
                                                                                   '-')):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    abds_of_one_condition[i] = np.std(abd.values)
        abds_of_all.append(abds_of_one_condition)
    with open('D:\\phd\\CytoCastNew/output/stds-M-Het.npy', 'wb') as f:
        np.save(f, abds_of_all)
    print(np.array(abds_of_all).shape)
    """
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        for i in range(len(IDs)):
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == 'M-Hom-' + region.split("\\")[3][:-4].replace('_',
                                                                                   '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    abds_of_one_condition[i] = np.mean(abd.values)
        abds_of_all.append(abds_of_one_condition)
    with open('D:\\phd\\CytoCastNew/output/stds-M-Hom.npy', 'wb') as f:
        np.save(f, abds_of_all)
    print(np.array(abds_of_all).shape)
    """
    for i in range(len(IDs)):
        if(abds_of_one_condition[i]>1):
            print(str(IDs[i])+":"+str(abds_of_one_condition[i]))
    """

    # print(exp.complexes.complexes.__getitem__(10).occurrences[0].abundance[0].condition)

def compact_print(d, indent=''):
    items = d.items()
    key, value = items[0]
    if not indent:
        print("{")

    if isinstance(value, dict):
        print(indent + "'{}': {{".format(key))
        compact_print(value, indent + ' ')
        print(indent + "'...'")
    else:
        print(indent + "'{}': '{}',".format(key, value))
        print(indent + "'...'")
    print(indent + "}")

def MultiPairwaiseTtest():
    from google.protobuf.json_format import MessageToDict
    import scipy.stats as stats
    exp = Experiment(fd.askopenfilenames(title="Experiment Binaries"))
    print([field.name for field in exp.complexes.DESCRIPTOR.fields])
    print([field.name for field in exp.complexes.complexes.__getitem__(0).DESCRIPTOR.fields])
    IDs = []
    i: int
    for i in range(len(exp.complexes.complexes)):
        IDs.append(exp.complexes.complexes.__getitem__(i).ID)
    # print(exp.complexes.complexes.__getitem__(10).ID)
    #print(MessageToDict(exp.complexes.complexes.__getitem__(10).occurrences[0]))
    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    colorsandtypes = colorsdata["colors"]
    colors = colorsandtypes[0]
    region_types_set = colorsandtypes[1]
    region_names = colorsandtypes[2]
    region_types = colorsandtypes[3]

    avgs=[]
    feature_importance = np.load(fd.askopenfilename(title="Feature Importance"))
    count=0
    for region in region_names:

        Ttest_of_one_condition = {}
        count+=1
        print(count)
        for i in range(len(IDs)):
            dict = {}
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == region.split("\\")[3][:-4].replace('_',
                                                                        '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    dict["WT"] = abd.values
                    break
            if "WT" not in dict:
                dict["WT"] = np.zeros(40)
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == 'm-hom-' + region.split("\\")[3][:-4].replace('_',
                                                                                   '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    dict["MT"] = abd.values
                    break
            if "MT" not in dict:
                dict["MT"] = np.zeros(40)
            test=stats.ttest_rel(dict["WT"], dict["MT"])
            if(not (np.isnan(test[0])  or np.isnan(test[1]))):
                Ttest_of_one_condition[i]=test
        #abds_of_all[region.split("\\")[3][:-4].replace('_','-')]=Ttest_of_one_condition;
            #abds_of_all.append(Ttest_of_one_condition)

        with open('D:\\phd\\CytoCastNew/output/Ttests/'+region.split("\\")[3][:-4].replace('_','-')+'.json', 'w') as outfile:
            json.dump(Ttest_of_one_condition, outfile)

        sum=0
        for key in Ttest_of_one_condition:
            sum+=feature_importance[key]*Ttest_of_one_condition[key][1]
        avg=sum/len(feature_importance)
        avgs.append(avg)

    figure1.clear()

    ax1=figure1.add_subplot(111)
    # evenly sampled time at 200ms intervals
    t = np.arange(0., len(region_names), 0.2)
    t2=np.arange(0., len(region_names), 1)

    # red dashes, blue squares and green triangles
    ax1.plot(t2,avgs, 'bs')
    ax1.axhline(y=.05, color='r', linestyle='dashed')
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()

    #with open('D:\\phd\\CytoCastNew/output/Ttests.npy', 'wb') as f:
    #    np.save(f, avgs)
    #print(np.array(abds_of_all).shape)


def shapiro(k:int):
    from scipy.stats import shapiro
    from google.protobuf.json_format import MessageToDict
    exp = Experiment(fd.askopenfilenames())
    print([field.name for field in exp.complexes.DESCRIPTOR.fields])
    print([field.name for field in exp.complexes.complexes.__getitem__(0).DESCRIPTOR.fields])
    IDs = []
    i: int
    for i in range(len(exp.complexes.complexes)):
        IDs.append(exp.complexes.complexes.__getitem__(i).ID)
    #print(len(IDs))
    #print(sorted(IDs)[-1])

    # print(exp.complexes.complexes.__getitem__(10).ID)
    #print(MessageToDict(exp.complexes.complexes.__getitem__(10).occurrences[0]))
    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    colorsandtypes = colorsdata["colors"]
    colors = colorsandtypes[0]
    region_types_set = colorsandtypes[1]
    region_names = colorsandtypes[2]
    region_types = colorsandtypes[3]
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        #for i in range(len(IDs)):
        for abd in exp.complexes.complexes.__getitem__(int(k)).occurrences[0].abundance:
            if (abd.condition == region.split("\\")[3][:-4].replace('_',
                                                                        '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                #abds_of_one_condition[i] = shapiro(abd.values)
                print(shapiro(abd.values)[1])
        #abds_of_all.append(abds_of_one_condition[1])
    #with open('D:\\phd\\CytoCastNew/output/stds-wild-type.npy', 'wb') as f:
    #    np.save(f, abds_of_all)
    #print(np.array(abds_of_all).shape)
    """
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        for i in range(len(IDs)):
            for abd in exp.complexes.complexes.__getitem__(i).occurrences[0].abundance:
                if (abd.condition == 'M-Het-' + region.split("\\")[3][:-4].replace('_',
                                                                                   '-')):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                    abds_of_one_condition[i] = np.std(abd.values)
        abds_of_all.append(abds_of_one_condition)
    with open('D:\\phd\\CytoCastNew/output/stds-M-Het.npy', 'wb') as f:
        np.save(f, abds_of_all)
    print(np.array(abds_of_all).shape)
    """
    abds_of_all = []
    for region in region_names:
        abds_of_one_condition = np.zeros(len(IDs))
        #for i in range(len(IDs)):
        for abd in exp.complexes.complexes.__getitem__(int(k)).occurrences[0].abundance:
            if (abd.condition == 'M-Hom-' + region.split("\\")[3][:-4].replace('_',
                                                                                   '-').lower()):  # 'M-Het-abd-data-H376.VIII.51-M1C'):
                #abds_of_one_condition[i] = shapiro(abd.values)
                print(shapiro(abd.values)[1])
        #abds_of_all.append(abds_of_one_condition[1])
    #with open('D:\\phd\\CytoCastNew/output/stds-M-Hom.npy', 'wb') as f:
    #    np.save(f, abds_of_all)
    #print(np.array(abds_of_all).shape)
    """
    for i in range(len(IDs)):
        if(abds_of_one_condition[i]>1):
            print(str(IDs[i])+":"+str(abds_of_one_condition[i]))
    """



def MultiPairwaiseTtest2():
    from google.protobuf.json_format import MessageToDict
    import scipy.stats as stats

    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    colorsandtypes = colorsdata["colors"]
    colors = colorsandtypes[0]
    region_types_set = colorsandtypes[1]
    region_names = colorsandtypes[2]
    region_types = colorsandtypes[3]

    avgs=[]
    avgs2=[]
    feature_importance = np.load(fd.askopenfilename(title="Feature Importance"))
    count=0
    for region in region_names:
        with open('D:\\phd\\CytoCastNew/output/Ttests/'+region.split("\\")[3][:-4].replace('_','-')+'.json') as f:
            Ttest_of_one_condition= json.load(f)
        sum=0
        sum2=0
        for key in Ttest_of_one_condition:
            if(int(key)==5):#ezt a komplexet nézem most
                sum+=Ttest_of_one_condition[key][1]#feature_importance[int(key)]*Ttest_of_one_condition[key][1]#
            sum2 += feature_importance[int(key)]*Ttest_of_one_condition[key][1]#
        avg=sum#/len(feature_importance)
        avg2=sum2#/len(feature_importance)
        avgs.append(avg)
        avgs2.append(avg2)


    indices=np.argsort(avgs) #Eztmentsd ki itt vannak az og regio indexek.
    sorted_avgs=np.sort(avgs)

    print("region bycomplex5 allaveraged")
    for i in range(len(indices)):
        print(str(indices[i])+" "+str(sorted_avgs[i])+" "+str(avgs2[indices[i]]))

    shanks = []
    gkaps=[]
    inputs = np.loadtxt("input2.csv",
                        delimiter=";", dtype=str)
    inputs = inputs.T
    for i in range(len(sorted_avgs)):
        #if(sorted_avgs[i]<0.05 and sorted_avgs[i]>0):
        # ax1.annotate(region_names[i].split("\\")[3][:-4].replace('_','-').split("-")[2]+'-'+region_names[i].split("\\")[3][:-4].replace('_','-').split("-")[3] , xy=(i, avgs[i]), xytext=(i-200, avgs[i]-.005),
        #           arrowprops=dict(arrowstyle="->", facecolor='black'))
        #print(indices[i])
        shanks.append(int(inputs[indices[i]][5]))
        gkaps.append(int(inputs[i][4]))

    with open('D:\\phd\\CytoCastNew/output/Ttests222333333.npy', 'wb') as f:
        np.save(f, avgs)
    figure1.clear()

    ax1=figure1.add_subplot(111)
    # evenly sampled time at 200ms intervals
    t = np.arange(0., len(region_names), 0.2)
    t2=np.arange(0., len(region_names), 1)
    color=[]
    with open('D:\\phd\\CytoCastNew/output/ttestpvaluessorted_shanks.txt', 'w') as f:
        for i in range(len(avgs)):
            #ax1.scatter(i, int(shanks[i]), region_types_set.index(region_types[indices[i]]),color=colors[region_types_set.index(region_types[indices[i]])])
            print(str(i)+' '+str(shanks[i])+' '+region_types_set[region_types_set.index(region_types[indices[i]])], file=f)

        #ax1.plot(t2,avgs, 'bs')
        """
            if(avgs[i]<0.05):
               color.append('red')
               print(str(gkaps[i]) + ' ' + str(shanks[i]) + ' ' + 'red', file=f)
            else:
                color.append('blue')
                print(str(gkaps[i]) + ' ' + str(shanks[i]) + ' ' + 'blue', file=f)
        """
    #ax1.scatter(gkaps, shanks, c=color)


    #ax1.axhline(y=.05, color='r', linestyle='dashed')
    #ax1.axvline(x=100, color='r', linestyle='dashed')
    ax1.yaxis.set_ticks(np.arange(0, 290, 20))
    ax1.xaxis.set_ticks(np.arange(0, 370, 20))
    plt.xlabel('GKAP abundance')#('GKAP abundance')#('Brain Regions sorted by p-values')
    plt.ylabel('Shank1 abundance')#('p-values')#

    import matplotlib.patches as mpatches
    pops = []
    for i in range(len(colors)):
        # legend
        pops.append(mpatches.Patch(color=colors[i], label=region_types_set[i]))

    import matplotlib.font_manager as font_manager

    font = font_manager.FontProperties(family='Arial',
                                       weight='normal',
                                       style='normal', size=12)
    figure1.legend(handles=pops, bbox_to_anchor=(-0.5, -0.3),
                   loc='center', ncol=5, numpoints=1, borderaxespad=0.1, prop=font)  # facecolor="plum"
    figure1.tight_layout()




    """
        if(sorted_avgs[i]>0.05):
            break
        
        if (avgs[i] > 0.04 and avgs[i] < 0.05):
            ax1.annotate(region_names[i].split("\\")[3][:-4].replace('_', '-').split("-")[2] + '-' +
                         region_names[i].split("\\")[3][:-4].replace('_', '-').split("-")[3], xy=(i, avgs[i]),
                         xytext=(i, avgs[i] - .03),
                         arrowprops=dict(arrowstyle="->", facecolor='black'))
    """
    figure1.canvas.figure.tight_layout()
    figure1.canvas.draw()


    #print(np.array(abds_of_all).shape)


def DistancesMut():

    WT=np.squeeze(np.load(fd.askopenfilename()))
    MUT = np.squeeze(np.load(fd.askopenfilename()))

    #plotDistances(calcDistances(WT,MUT,folder_selected_c+'/'),folder_selected_c+'/')
    cos_sims=[]
    print(len(WT))
    for i in range(len(WT)):
        cos_sim = np.dot(WT[i], MUT[i]) / (np.linalg.norm(WT[i]) * np.linalg.norm(MUT[i]))
        cos_sims.append(cos_sim)
    import matplotlib.pyplot as plt

    #plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()

    ax.plot(range(len(WT)), cos_sims)

    #ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
    #       ylim=(0, 8), yticks=np.arange(1, 8))
    ax.set_xlabel('Brain regions')
    ax.set_ylabel('cosine correlation');
    ax.axhline(0.8, ls='--', color='r')
    ax.annotate('H376.VIII.53_MD (360)', xy=(360, 0.52), xytext=(370, 0.7),
                arrowprops=dict(arrowstyle="->",facecolor='black'))
    ax.annotate('H376.V.53_MD (270)', xy=(270, 0.77), xytext=(100, 0.65),
                arrowprops=dict(arrowstyle="->", facecolor='black'))
    ax.annotate('H376.VIII.52_STR (352)', xy=(352, 0.788), xytext=(200, 0.6),
                arrowprops=dict(arrowstyle="->", facecolor='black'))
    #ax.title('')
    plt.show()

def splitMatrixbyType():
    means=np.load(fd.askopenfilename(title="means"))
    from tkinter import filedialog
    folder_selected = filedialog.askdirectory()
    with open("D:\phd\CytoCastNew/colors.json") as f:
        colorsdata = json.load(f)
    for type in colorsdata["colors"][1]:
        print(type)
        type_array=[]
        for i in range(len(colorsdata["colors"][2])):
            if (type in colorsdata["colors"][2][i]):
                type_array.append(means[i])
        with open(folder_selected+"/"+type+'.npy', 'wb') as f:
            np.save(f, type_array)

def readPvalues(k):
    import glob
    files = sorted(glob.glob("D:\phd\CytoCastNew/output/Ttests/*"))
    for file in files:
        with open(file, 'r') as file_:
            data = json.load(file_)
            if "5" in data.keys():
            # Feltételezzük, hogy az adatok lista formátumban vannak
                print(file +";" + str(data["5"][1]))
            else:
                print(file +";0")

def main():
    global exp
    # Adding light and dark mode images

    root = Tk()

    style = ttk.Style(root)
    style.theme_use('alt')
    style.theme_use('alt')
    style.configure('TFrame', foreground='green', background='#B7B7B7')
    root.configure(background='#B7B7B7')
    style.configure('TLabel', foreground='black', background='#B7B7B7')

    # root.geometry("1000x1000")
    root.attributes("-fullscreen", True)
    frame = ttk.Frame(root)
    X = frame.winfo_screenwidth()
    Y = frame.winfo_screenheight()
    frame.pack()

    leftframe = ttk.Frame(root, width=round(X * .5), height=800)
    leftframe.pack(side=LEFT)

    rightframe = ttk.Frame(root, width=round(X * .5), height=800)
    rightframe.pack(side=RIGHT)

    btn_load = ttk.Button(leftframe, text="Load Data",
                          command=lambda: load())
    btn_load.place(x=10, y=10)
    t = Text(rightframe, highlightcolor='blue', highlightbackground='brown', fg='white', bg='#191A19')

    t.place(x=0, y=300)
    # create instance of file like object
    pl = PrintLogger(t)

    # replace sys.stdout with our object
    sys.stdout = pl
    global figure1
    figure1 = plt.figure()  # figsize=(10, 10))
    # global ax1

    bar1 = FigureCanvasTkAgg(figure1, leftframe)
    toolbar = NavigationToolbar2Tk(bar1, leftframe)
    toolbar.config(background="#B7B7B7")
    toolbar._message_label.config(background="#B7B7B7")
    toolbar.place(x=200, y=0)
    toolbar.update()
    bar1.get_tk_widget().place(x=50, y=50)
    """
    # Dropdown menu options
    options = [
        "Wild Type",
        "Homo",
        "Hetero"
    ]
    # datatype of menu text
    clicked = StringVar()
    # initial menu text
    clicked.set("Wild Type")
    # Create Dropdown menu
    drop = ttk.OptionMenu(rightframe, clicked, *options)
    drop.place(x=10, y=200)
    """
    light = PhotoImage(file="D:\phd\CytoCastNew/sun.png")
    dark = PhotoImage(file="D:\phd\CytoCastNew/moon.png")

    btn = ttk.Button(rightframe, image= dark,text="Sample", command=lambda: change_theme(style, root, toolbar))
    btn.place(x=200, y=0)

    btn_heatmap = ttk.Button(rightframe, text="HeatMap",
                             command=lambda: heatmap(fd.askopenfilename(),
                                                     fd.askopenfilename())
                             )
    btn_heatmap.place(x=10, y=10)
    btn_heatmap = ttk.Button(rightframe, text="Distances",
                             command=lambda: DistancesMut()
                             )
    btn_heatmap.place(x=100, y=10)
    btn_pca = ttk.Button(rightframe, text="PCA",
                         command=lambda: pca(fd.askopenfilename(),
                                             fd.askopenfilename(), ))
    btn_pca.place(x=10, y=60)

    label = ttk.Label(rightframe, text="k", font=('Arial', 12))
    label.place(x=200, y=40)
    label = ttk.Label(rightframe, text="random state", font=('Arial', 12))
    label.place(x=300, y=40)
    ent_k = ttk.Entry(rightframe, width=12)
    ent_k.place(x=200, y=60)
    ent_rnd_state = ttk.Entry(rightframe, width=12)
    ent_rnd_state.place(x=300, y=60)

    btn_pca = ttk.Button(rightframe, text="K-means",
                         command=lambda: kMeans(np.load(fd.askopenfilename()),
                                                int(ent_k.get()), int(ent_rnd_state.get())))

    btn_pca.place(x=100, y=60)
    btn_pca = ttk.Button(rightframe, text="readpvalues",
                         command=lambda: readPvalues(0))
    btn_pca.place(x=200,y=270)
    btn_pca = ttk.Button(rightframe, text="Complex",
                         command=lambda: complex(int(ent_k.get())))
    btn_pca.place(x=10, y=180)
    btn_pca = ttk.Button(rightframe, text="Calc sizes",
                         command=lambda: complexSize())
    btn_pca.place(x=10, y=220)
    btn_pca = ttk.Button(rightframe, text="Plot sizes",
                         command=lambda: plotComplexSize(int(ent_k.get())))
    btn_pca.place(x=100, y=220)

    agreement=True
    var = IntVar()
    chk_bar=ttk.Checkbutton(rightframe,
                    text='Bar Chart', variable=var
                    ).place(x=100,y=110)
    isAll=False
    chk_all = ttk.Checkbutton(rightframe,
                              text='All',
                              # command=agreement_changed,
                              variable=isAll,
                              onvalue=True,
                              offvalue=False).place(x=200, y=110)
    text_region1=StringVar()
    text_region2=StringVar()
    ent_region1 = ttk.Entry(rightframe, width=10, textvariable=text_region1).place(x=250,y=110)
    ent_region2 = ttk.Entry(rightframe, width=10,textvariable=text_region2).place(x=350,y=110)

    btn_pie = ttk.Button(rightframe, text="Plot Pies",
                         command=lambda: plotAbs(fd.askopenfilename(),fd.askopenfilename(),[int(text_region1.get()),int(text_region2.get())],1,var.get()))#451,519
    btn_pie.place(x=10, y=110)
    btn_pie = ttk.Button(rightframe, text="T test",
                         command=lambda:MultiPairwaiseTtest())
    btn_pie.place(x=10, y=140)
    btn_pie = ttk.Button(rightframe, text="T test 2",
                         command=lambda: MultiPairwaiseTtest2())
    btn_pie.place(x=100, y=140)
    btn_pie = ttk.Button(rightframe, text="Shapiro",
                         command=lambda: shapiro(ent_k.get()))
    btn_pie.place(x=100, y=180)
    btn_pie = ttk.Button(rightframe, text="Feature",
                         command=lambda: featureImportance())
    btn_pie.place(x=200, y=140)
    btn_pie = ttk.Button(rightframe, text="Split Matrix",
                         command=lambda: splitMatrixbyType())
    btn_pie.place(x=200, y=170)
    """
    switch = ttk.Button(rightframe, image=light
                    )
    switch.place(x=10, y=170)
    """

    root.bind('<Escape>', lambda e: root.quit())
    root.title("CytoCast PostProcessing")
    root.mainloop()
    return 0


if __name__ == "__main__":
    main()

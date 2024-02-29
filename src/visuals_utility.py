import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
import color_utility
import textwrap
import matplotlib.ticker as ticker

def single_binary_confusion_matrix(conf_matrix, title=None):
    '''
        Plot a single confustion matrix whose target is binary.
    '''
    # Create a custom colormap using ListedColormap
    colors = ['#FAA0A0','#AFE1AF']
    custom_cmap = ListedColormap(colors)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=custom_cmap, cbar=False)

    # Add labels to each box
    plt.text(0.5, 0.35, 'True -ve', color='green', horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 0.35, 'False +ve', color='darkred', horizontalalignment='center', verticalalignment='center')
    plt.text(0.5, 1.35, 'False -ve', color='darkred', horizontalalignment='center', verticalalignment='center')
    plt.text(1.5, 1.35, 'True +ve', color='green', horizontalalignment='center', verticalalignment='center')

    plt.title(f"{(title if title else "")} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def multiple_binary_confusion_matrices(conf_matrices):
    '''
    
    '''
    colors = ['#FAA0A0','#AFE1AF']
    custom_cmap = ListedColormap(colors)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))

    for i, (m_name, cm) in enumerate(conf_matrices.items()):
        row = i // 2  
        col = i % 2   
        ax = axes[row, col] 
        sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap, cbar=False, ax=ax)

        ax.set_title(f'Confusion Matrix - {m_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # Add labels to each box
        ax.text(0.5, 0.35, 'True -ve', color='green', horizontalalignment='center', verticalalignment='center')
        ax.text(1.5, 0.35, 'False +ve', color='darkred', horizontalalignment='center', verticalalignment='center')
        ax.text(0.5, 1.35, 'False -ve', color='darkred', horizontalalignment='center', verticalalignment='center')
        ax.text(1.5, 1.35, 'True +ve', color='green', horizontalalignment='center', verticalalignment='center')

        ax.xaxis.set_ticklabels(['0', '1'])
        ax.yaxis.set_ticklabels(['0', '1'])
    
    plt.tight_layout()
    plt.show()


def single_multi_confusion_matrix(conf_matrix, clabels, title=None):
    '''
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

    plt.xticks(ticks=range(len(clabels)), labels=clabels, rotation=0)
    plt.yticks(ticks=range(len(clabels)), labels=clabels, rotation=0)

    plt.title(f"{title if title else ""} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def show_accuracy_scores(models_accuracy):
    '''
    '''
    models = list(models_accuracy.keys())
    accuracy_scores = list(models_accuracy.values())

    color_palette = color_utility.color_palettes['star_ratings'][:len(models)]

    plt.figure(figsize=(10, 6))
    bars = sns.barplot(y=models, x=accuracy_scores, hue=models, palette=color_palette)

    # print accuracy value at the end of bar.
    for bar, score in zip(bars.patches, accuracy_scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{score:.2f}', ha='left', va='center')

    #plt.yticks(fontsize=10)
    ytick_labels = [textwrap.fill(label.get_text(), width=12) for label in plt.gca().get_yticklabels()]
    plt.gca().set_yticklabels(ytick_labels)

    plt.xlim(right=1.0)

    plt.title('Accuracy Scores of Models')

    plt.xlabel('Accuracy Score')
    plt.ylabel('Model')

    plt.show()
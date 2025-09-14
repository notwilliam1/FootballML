import matplotlib.pyplot as plt
import seaborn as sns

def create_visuals(data, cm, accuracy):

    print("\n" + "=" * 40)
    print("Creating Visuals")
    print("=" * 40)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))

    # Plot 1: Distribution of success lvls
    success_counts = data['Team_Success'].value_counts().sort_index()
    bars = axes[0].bar(['Poor', 'Good', 'Excellent'], success_counts.values,
                       color=['red', 'orange', 'green'], alpha = 0.7)
    axes[0].set_title("Distribution of Team Success Levels")
    axes[0].set_ylabel("Number of Teams")

    # Add numbers on top of bars
    for bar, count in zip(bars, success_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  str(count), ha = 'center', va = 'bottom')
        
    # Plot 2: Confusion matrix heatmap
    sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', ax = axes[1],
                xticklabels = ['Poor', 'Good', 'Excellent'],
                yticklabels = ['Poor', 'Good', 'Excellent'])
    axes[1].set_title(f"Confusion Matrix (Accuracy: {accuracy:.3f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.show()

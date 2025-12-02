import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from skimage import morphology, measure
import tifffile

def create_black_background_colormap():
    """
    Create a colormap with a black background
    
    Returns:
    --------
    custom_cmap : ListedColormap
        Colormap with the first color set to black
    
    Notes:
    ------
    Uses the 'tab20' colormap as a base and sets the first color to black.
    Useful for visualizing labeled images with a dark background.
    """
    base_cmap = plt.colormaps['tab20']
    colors = base_cmap(np.linspace(0, 1, 20))
    colors[0] = [0, 0, 0, 1] 
    return ListedColormap(colors)

def generate_cytoplasmic_ring(nucs, ring_width=2):
    """
    Generate cytoplasmic ring from nuclear labels
    
    Parameters:
    -----------
    nucs : list
        Nuclear label images
    ring_width : int, optional
        Width of the cytoplasmic ring (default: 2)
    
    Returns:
    --------
    cytos : list
        Cytoplasmic ring images
    
    Notes:
    ------
    Creates cytoplasmic rings by dilating nuclear labels 
    and subtracting the original nuclear image.
    Saves the result as a TIFF file.
    """
    cytos = [morphology.dilation(img, morphology.disk(ring_width)) - img for img in nucs]
    tifffile.imwrite('cyto.tif', cytos)
    
    return cytos

def extract_intensity_features(nucs, cytos, imgs):
    """
    Extract intensity features from nuclear and cytoplasmic labels
    
    Parameters:
    -----------
    nucs : list
        Nuclear label images
    cytos : list
        Cytoplasmic ring label images
    imgs : ndarray
        Original image stack
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with intensity features for each cell and timepoint
    
    Notes:
    ------
    Calculates mean intensities, area, and cytoplasmic/nuclear (C/N) ratio.
    Drops rows with NaN values to ensure data quality.
    """
    df = pd.DataFrame()
    for i, (nuc, cyto, img) in enumerate(zip(nucs, cytos, imgs)):
        prop1 = pd.DataFrame(measure.regionprops_table(
            label_image=nuc.astype('int'), 
            intensity_image=img[0], 
            properties=['label', 'intensity_mean', 'area']
        ))
        prop1.columns = ['label', 'nuc_intensity', 'area']
        
        prop2 = pd.DataFrame(measure.regionprops_table(
            label_image=cyto.astype('int'), 
            intensity_image=img[0], 
            properties=['label', 'intensity_mean']
        ))
        prop2.columns = ['label', 'cyto_intensity']
        prop2 = prop2.drop(columns=['label'])
        
        prop = pd.concat([prop1, prop2], axis=1)
        prop['time'] = np.full(len(prop), i)
        df = pd.concat([df, prop], axis=0)
    
    df['C/N'] = df['cyto_intensity'] / df['nuc_intensity']
    df = df.dropna(how='any')
    df['label'] = df['label'].values.astype('int')
    df = df.reset_index(drop=True)
    
    return df

def visualize_cn_ratio(df):
    """
    Visualize Cytoplasmic/Nuclear (C/N) ratio across timepoints
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing C/N ratio data
    
    Notes:
    ------
    Creates three subplots:
    1. Mean C/N ratio with standard deviation
    2. Individual cell C/N ratio trajectories
    3. Heatmap of C/N ratio sorted by mean value
    """
    df_p = df.pivot(index='label', columns='time', values='C/N')
    df_p['mean'] = df_p.mean(axis=1)
    df_p = df_p.sort_values('mean')
    df_p = df_p.drop(columns=['mean'])
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.lineplot(data=df, x='time', y='C/N', errorbar='sd')
    plt.ylim(0.2, 2)
    plt.title('C/N Ratio Over Time (Mean ± SD)')
    
    plt.subplot(1, 3, 2)
    sns.lineplot(data=df, x='time', y='C/N', hue='label', legend=False)
    plt.ylim(0.2, 2)
    plt.title('C/N Ratio Over Time (Individual Cells)')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(df_p, vmin=0.2, vmax=2, cmap='viridis')
    plt.title('C/N Ratio Heatmap')
    
    plt.tight_layout()
    plt.show()
    
def visualize_cn_ratio_timelapse(
    label_nuc_series, df, 
    cn_ratio_column='CN_ratio', 
    time_column='Time', 
    cmap='bwr', 
    vmin=None, 
    vmax=None, 
    save_path=None, 
    dpi=300,
    fps=1,
    interval=1000,
    background_color='black'
):
    """
    Visualize C/N ratio across timepoints as a timelapse
    
    Parameters:
    -----------
    label_nuc_series : list
        Nuclear label images for each timepoint
    df : pandas.DataFrame
        DataFrame with timepoint and C/N ratio information
    cn_ratio_column : str, optional
        Column name for C/N ratio (default: 'CN_ratio')
    time_column : str, optional
        Column name for timepoints (default: 'Time')
    cmap : str, optional
        Colormap for visualization (default: 'bwr')
    vmin, vmax : float, optional
        Color scaling limits (default: computed from data)
    save_path : str, optional
        Path to save the output image/animation
    dpi : int, optional
        Resolution for saved image (default: 300)
    fps : float, optional
        Frames per second for animation (default: 1)
    interval : int, optional
        Milliseconds between frames (default: 1000)
    background_color : str, optional
        Background color for the visualization (default: 'black')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    anim : matplotlib.animation.FuncAnimation
    
    Notes:
    ------
    Creates a timelapse visualization of C/N ratio with:
    - Dark background
    - Colorbar
    - Time-dependent C/N ratio mapping
    Saves both static image and animated GIF.
    """
    # Time point extraction
    unique_times = sorted(df[time_column].unique())
    
    # Visualization preparation
    if vmin is None:
        vmin = df[cn_ratio_column].quantile(0.01)
    if vmax is None:
        vmax = df[cn_ratio_column].quantile(0.99)
    
    def create_cn_ratio_image(time_point, label_nuc):
        """
        Create C/N ratio image for a specific timepoint
        
        Parameters:
        -----------
        time_point : float
            Specific timepoint
        label_nuc : ndarray
            Nuclear label image
        
        Returns:
        --------
        cn_ratio_image : ndarray
            C/N ratio mapped image
        """
        # Extract data for the specific timepoint
        time_df = df[df[time_column] == time_point]
        
        # Create C/N ratio image
        cn_ratio_image = np.zeros_like(label_nuc, dtype=float)
        cn_ratio_image[label_nuc == 0] = np.nan  # Background as NaN
        
        for _, row in time_df.iterrows():
            label_id = row['label']
            cn_ratio = row[cn_ratio_column]
            cn_ratio_image[label_nuc == label_id] = cn_ratio
        
        return cn_ratio_image
    
    def create_animation():
        # Set dark background style
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 8), 
                                facecolor=background_color,
                                edgecolor='none')
        
        # First image display
        first_time = unique_times[0]
        first_cn_ratio_img = create_cn_ratio_image(first_time, label_nuc_series[0])
        im = ax.imshow(first_cn_ratio_img, 
                       cmap=cmap, 
                       vmin=vmin, 
                       vmax=vmax,
                       interpolation='nearest')
        
        # Title and axis
        title = ax.set_title(f'C/N Ratio: Time {first_time}', 
                              color='white',
                              fontsize=12)
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, 
                            aspect=30,
                            pad=0.08)
        cbar.set_label('C/N ratio', 
                       rotation=270, 
                       labelpad=20, 
                       color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')
        
        # Animation update function
        def update(frame):
            time_point = unique_times[frame]
            cn_ratio_img = create_cn_ratio_image(time_point, label_nuc_series[frame])
            im.set_array(cn_ratio_img)
            title.set_text(f'C/N Ratio: Time {time_point}')
            return [im, title]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, 
            frames=len(unique_times), 
            interval=interval,
            blit=True
        )
        
        return fig, anim
    
    # Create and save animation
    fig, anim = create_animation()
    
    # Save
    if save_path:
        # Save static image
        static_save_path = save_path.replace('.gif', '_static.png')
        plt.figure(fig.number)
        plt.tight_layout()
        fig.savefig(static_save_path, dpi=dpi, bbox_inches='tight')
        print(f"Static image saved to {static_save_path}")
        
        # Save as GIF animation
        gif_save_path = save_path if save_path.endswith('.gif') else save_path + '.gif'
        anim.save(gif_save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {gif_save_path}")
    
    plt.close(fig)  # Memory management
    
    return fig, anim
# language-networks

Designed to be imported from python. 
```
import plots 

data_path="/path/to/directory/with/embeddings//"
out_path="/path/to/outputs//"

plots.run_small_samp(data_path, out_path, sampsize=.01)
plots.plot_histograms(data_path, out_path)

plots.run_small_samp2(data_path, out_path, sampsize=.01)
plots.plot_violin(data_path, out_path, metric='cosine')
plots.plot_violin(data_path, out_path, metric='euclidean')
```
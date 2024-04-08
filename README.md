# DCASE SELD Visualizer

<img src="https://github.com/adrianSRoman/dcase_seld_visualizer/blob/main/assets/ground_truth.gif" width="900" height="280"/>

## Execute as

```Shell
python visualizer.py
```

## Visualize the output as a GIF

```shell
ffmpeg -framerate 10 -pattern_type glob -i 'output/*.jpg' -vf "fps=10" output_event.gif
```

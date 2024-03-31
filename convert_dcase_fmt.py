import csv
import math

def cartesian_to_polar(x, y, z):
    # Calculate azimuth
    azimuth = math.atan2(y, x)
    azimuth = math.degrees(azimuth)
    azimuth = (azimuth + 360) % 360  # Normalize to [0, 360] range
    if azimuth > 180:
        azimuth -= 360  # Adjust to [-180, 180] range
    
    # Calculate elevation
    distance_xy = math.sqrt(x**2 + y**2)
    elevation = math.atan2(z, distance_xy)
    elevation = math.degrees(elevation)
    
    # Calculate distance (set to 0)
    distance = 0
    
    return azimuth, elevation, distance

def main(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        with open(output_file, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                x, y, z = map(float, row[3:6])  # Extract Cartesian coordinates
                azimuth, elevation, distance = cartesian_to_polar(x, y, z)
                frame_number, active_class_index, source_number_index = map(int, row[:3])
                writer.writerow([frame_number, active_class_index, source_number_index, int(azimuth), int(elevation), int(distance)])

if __name__ == "__main__":
    input_file = "fold5_room1_mix003_pred.csv"
    output_file = "fold5_room1_mix003_pred_polar.csv"
    main(input_file, output_file)


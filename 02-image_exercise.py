#%%
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


animal_dir = Path("Images/")
animal_files = list(animal_dir.glob("/*.png"))
for animal_file in animal_files:
    animal_name = animal_file.stem
    parent_directory = animal_file.parent.name
    print(f"Animal {animal_name} is in {parent_directory}")

    animal_image = Image.open(str(animal_file))
    print(f"The animal image for {animal_name} is size {animal_image.size}")
    animal_array = np.array(animal_image)
    bw_animal_array = np.average(animal_array, axis=2)
    bw_animal_img = Image.fromarray(bw_animal_array)

    # Show it
    plt.imshow(bw_animal_img)
    plt.show()

    new_animal_img = Image.fromarray(bw_animal_array.astype(np.unit8))
    Path(f"Images/monochrome/{parent_directory}").mkdir(exist_ok=True, parents=True)
    new_animal_file_path = Path(f"Images/monochrone/{parent_directory}/{animal_name}")
    new_animal_img.save(new_animal_file_path, "PNG")
#%%
# Paths
# Fill this according to own setup
BACKGROUND_DIR = 'E:/Source/EffortlessCV/data/office/'
BACKGROUND_GLOB_STRING = '*.jpg'
POISSON_BLENDING_DIR = 'pb'
SELECTED_LIST_FILE = 'demo_data_dir/selected.txt'
DISTRACTOR_LIST_FILE = 'demo_data_dir/neg_list.txt' 
DISTRACTOR_DIR = 'E:/Source/EffortlessCV/data/objects/'
DISTRACTOR_GLOB_STRING = '*_mask.jpg'
INVERTED_MASK = True # Set to true if white pixels represent background

# Parameters for generator
NUMBER_OF_WORKERS = 10
BLENDING_LIST = ['gaussian']

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 1

MIN_NO_OF_DISTRACTOR_OBJECTS = 0
MAX_NO_OF_DISTRACTOR_OBJECTS = 0
WIDTH = 640
HEIGHT = 480
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
MIN_SCALE = 0.1 # min scale for scale augmentation
MAX_SCALE = 0.5 # max scale for scale augmentation
MAX_DEGREES = 5 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
MIN_WIDTH = 100 # Minimum width of object to use for data generation
MIN_HEIGHT = 100 # Minimum height of object to use for data generation

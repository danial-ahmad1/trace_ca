import argparse, csv, logging, os, re, sys, subprocess, time
from datetime import datetime
from dataclasses import dataclass

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger('uSIM-CA CRON Runner')
logger.setLevel(LOGLEVEL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a file handler
file_handler = logging.FileHandler('logs/CAPCronRunner.log')
file_handler.setLevel(LOGLEVEL)
file_handler.setFormatter(formatter)
# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGLEVEL)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

@dataclass
class Device:
    """Class for keeping track of information about a device."""
    id: str
    has_ims: bool = False
    has_txt: bool = False

    def is_valid(self) -> bool:
        return self.has_ims and self.has_txt


def get_usimca_directories(path, mtime_threshold=0):
    pattern = re.compile(r"^usimca_\d{8}_\d{4}$", re.ASCII) # Match usimca_YYYYMMDD_HHMM"
    with os.scandir(path) as it:
        directories = [entry.name for entry in it if pattern.match(entry.name) and entry.is_dir() and 
                       time.time() - entry.stat().st_mtime >= mtime_threshold]
    return directories



def get_metadata_file(path):
    # ensure device_metadata.csv file exists within a path
    
    device_metadata = os.path.join(path, 'device_metadata.csv');
    
    if not os.path.isfile(device_metadata):
        logger.error(device_metadata + ' is missing.')
        return None

    logger.info('Device metadata found at ' + device_metadata)
    return device_metadata


def get_device_info(path):
    # Collate ims and txt by device ID
    pattern = re.compile(r"^CA\d{4}_.*[ims|txt]$", re.ASCII) # Match ims and txt files that start with valid device ID"
    with os.scandir(path) as it:
        device_info = {}
        for entry in it:
            if not entry.is_file() or not pattern.match(entry.name):
                continue
                
            device_id = entry.name.split("_")[0]
            device = device_info.get(device_id, Device(device_id))
            match os.path.splitext(entry.name)[1][-3:]:
                case "ims":
                    device.has_ims = True
                case "txt":
                    device.has_txt = True
                
            device_info.update({device_id: device})

    return device_info


def check_directory(metadata_file, device_info):
    is_valid = True
    
    # Use device_metadata.csv to validate device image files exist
    with open(metadata_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'Device Id' not in reader.fieldnames:
            logger.error('device_metadata is missing "Device Id" column.')
            return False

        for row in reader:
            device_id = row.get('Device Id')
            if device_id is None:
                logger.error('"Device Id" value is blank.')
                is_valid = False
                continue
                
            device = device_info.get(device_id)
            if device is None:
                logger.error(device_id + ' has no image files.')
                is_valid = False
                continue
                
            if not device.is_valid():
                logger.error(device_id + ' is missing files.')
                is_valid = False
                continue
                
    return is_valid


def validate_directory(path):

    logger.info('Checking source directory: ' + path)
    metadata_file = get_metadata_file(path)
    if not metadata_file:
        return False
    
    device_info = get_device_info(path)
    for device in device_info.values():
        logger.debug('Device info: ' + str(device))

    return check_directory(metadata_file, device_info)
    

def process_directory(dir_name, source_dir, dest_dir):
    dir_path = os.path.join(source_dir, dir_name)
    logger.info('Processing ' + dir_path)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    result = subprocess.run(['python', 'RunCAPNode.py', '-f', dir_name], capture_output=True, text=True)

    if result.returncode == 0:
        logger.info("Successfully processed " + dir_path)
    else:
        logger.error("Failed to process " + dir_path)

    if result.stdout:
        #write stdout to RunCAPNode_{dir_name}_{timestamp}.out
        out_filename = "logs/RunCAPNode_{}_{}.out".format(dir_name, timestr)
        with open(out_filename, "w") as out_f:
            out_f.write(result.stdout)

        logger.info("stdout logged to " + out_filename)
            
    if result.stderr:
        #write stderr to RunCAPNode_{dir_name}_{timestamp}.err
        err_filename = "logs/RunCAPNode_{}_{}.err".format(dir_name, timestr)
        with open(err_filename, "w") as err_f:
            err_f.write(result.stderr)

        logger.info("stderr logged to " + err_filename)
            
    
def main(args):
    
    source = get_usimca_directories(args.source_dir, 300)
    dest = get_usimca_directories(args.dest_dir)
    new = [directory for directory in source if directory not in dest]

    if not new:
        logger.info("No new directories to process")

    for directory in new:
        dir_path = os.path.join(args.source_dir, directory)
        if not validate_directory(dir_path):
            logger.info(dir_path + ' is not valid. skipping')
            continue

        process_directory(directory, args.source_dir, args.dest_dir)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='source_dir', type=str, 
                        help="The directory to scan for new data acquisition run directories")
    parser.add_argument(dest='dest_dir', type=str, 
                        help="The directory to check for existing processed results")
    parser.add_argument(dest='mtime', type=int, 
                        help="How long the data acquisition run directories should be silent before trying to process it. Specified in seconds.")

    main(parser.parse_args())

    # Set exit status code if any error messages were logged
    if 40 in logger._cache and logger._cache[40]:
        sys.exit(40)




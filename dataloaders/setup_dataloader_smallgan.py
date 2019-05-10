import glob
from .ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import  DataLoader

def setup_dataloader(name,h=128,w=128,batch_size=4,num_workers=4,ext='png'):
    '''
    instead of setting up dataloader that read raw image from file, 
    let's use store all images on cpu memmory
    because this is for small dataset
    '''
    img_path_list = glob.glob(f"./data/{name}/*.{ext}")
        
    assert len(img_path_list) > 0

    transform = transforms.Compose([
            transforms.Resize( min(h,w) ),
            transforms.CenterCrop( (h,w) ),
            transforms.Grayscale(num_output_channels=3), # remove me later
            transforms.ToTensor(),
            ])
    
    img_path_list = [[path,i] for i,path in enumerate(sorted(img_path_list))]
    dataset = ImageListDataset(img_path_list,transform=transform)
    
    return  DataLoader([data for data in  dataset],batch_size=batch_size, 
                            shuffle=True,num_workers=num_workers)
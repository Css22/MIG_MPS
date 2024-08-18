from jobs.entry import get_input, get_model

def warmup(task, batch):
    if task == 'bert':  
        model = get_model(task)
        model = model().half().cuda(0).eval()
    else:
        model = get_model(task)
        model = model().cuda(0).eval()

    if task == 'bert':
        input,masks = get_input(task, batch)
    else:
        input = get_input(task, batch)
    
    if task == 'bert':
        output= model.run(input,masks,0,12).cpu()
    elif task == 'deeplabv3':
        output= model(input)['out'].cpu()
    else:
        output=model(input).cpu() 


if __name__ == "__main__":
    warmup(task='resnet50', batch=32)
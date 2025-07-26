from model.bvp import *
import yaml
import shutil
import argparse 
   

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--c', type=str, default='configs/config_example.yaml')

    args = args.parse_args()
    pipe = load_pipe('cuda:0')

    config_bvp = yaml.safe_load(open(args.config_file))
    
    if not os.path.exists('results'):
        os.mkdir('results/')

    out_dir = config_bvp['output_args']['out_dir']
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    shutil.copy(args.config_file, out_dir)

    imgA = Image.open(config_bvp['pathA'])
    imgB = Image.open(config_bvp['pathB'])

    bvp_solver = Geodesic_BVP(pipe, imgA=imgA, imgB=imgB, **config_bvp)
    print('Start solving BVP')
    bvp_solver.solve()
    print('Done')
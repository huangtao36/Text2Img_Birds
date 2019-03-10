import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, default='/home/OpenResource/Datasets/Caltech200_birds/CUB_200_2011/images', help='root directory that contains images')
    parser.add_argument('--caption_root', type=str, default='/home/OpenResource/Datasets/Caltech200_birds/cub_icml', help='root directory that contains captions')
    parser.add_argument('--trainclasses_file', type=str, default='trainvalclasses.txt', help='text file that contains training classes')
    parser.add_argument('--fasttext_model', type=str, default='/home/OpenResource/PreTrainModel/wiki_en.bin', help='pretrained fastText model (binary file)')
    parser.add_argument('--save_filename', type=str, default='/models/text_embedding_birds.pth', help='checkpoint file')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads for fetching data (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='number of threads for fetching data (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='learning rate (dafault: 0.0002)')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='margin for pairwise ranking loss (default: 0.2)')
    parser.add_argument('--embed_ndim', type=int, default=300,
                        help='dimension of embedded vector (default: 300)')
    parser.add_argument('--max_nwords', type=int, default=50,
                        help='maximum number of words (default: 50)')


    args = parser.parse_args()

    return args

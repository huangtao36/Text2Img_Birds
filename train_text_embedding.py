from pyfasttext import FastText
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from model import VisualSemanticEmbedding
from data import ReedICML2016
from config import get_config

args = get_config()
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def pairwise_ranking_loss(margin, x, v):
    zero = torch.zeros(1)
    diag_margin = margin * torch.eye(x.size(0))
    zero, diag_margin = zero.to(DEVICE), diag_margin.to(DEVICE)

    x = x / torch.norm(x, 2, 1, keepdim=True)
    v = v / torch.norm(v, 2, 1, keepdim=True)
    prod = torch.matmul(x, v.transpose(0, 1))
    diag = torch.diag(prod)
    for_x = torch.max(
        zero, margin - torch.unsqueeze(diag, 1) + prod) - diag_margin
    for_v = torch.max(
        zero, margin - torch.unsqueeze(diag, 0) + prod) - diag_margin
    return (torch.sum(for_x) + torch.sum(for_v)) / x.size(0)


if __name__ == '__main__':

    args.img_root = '/home/OpenResource/Datasets/Caltech200_birds/CUB_200_2011/images'
    args.caption_root = '/home/OpenResource/Datasets/Caltech200_birds/cub_icml'
    args.trainclasses_file = 'trainvalclasses.txt'
    args.fasttext_model = '/home/OpenResource/PreTrainModel/wiki_en.bin'
    args.save_filename = './models/text_embedding_birds.pth'

    print('Loading a pretrained fastText model...')
    word_embedding = FastText(args.fasttext_model)

    print('Loading a dataset...')
    train_data = ReedICML2016(
        args.img_root,
        args.caption_root,
        args.trainclasses_file,
        word_embedding,
        args.max_nwords,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))

    word_embedding = None  # 占内存，释放掉

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads)

    model = VisualSemanticEmbedding(args.embed_ndim, device=DEVICE)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        avg_loss = 0
        for i, (img, desc, len_desc) in enumerate(train_loader):

            img = img.to(DEVICE)
            desc = desc.to(DEVICE)
            len_desc, indices = torch.sort(len_desc, 0, True)
            indices = indices.numpy()
            img = img[indices, ...]
            desc = desc[indices, ...].transpose(0, 1)
            desc = nn.utils.rnn.pack_padded_sequence(
                desc, len_desc.numpy())

            optimizer.zero_grad()

            img_feat, txt_feat = model(img, desc)
            # args.margin = 0.2
            loss = pairwise_ranking_loss(
                args.margin, img_feat, txt_feat)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                      % (epoch + 1,
                         args.num_epochs,
                         i + 1, len(train_loader), avg_loss / (i + 1)))

        torch.save(model.state_dict(), args.save_filename)

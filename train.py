import argparse
import json
import os

import torchaudio
import wandb

import dog_net as net
import torch
from torch import optim

from dog_dataset import get_dataloader
from dog_utils import get_mask_from_lengths


def Train(models, epochs, train_loader, test_loader, optimizers, device, model_dir, log_path, config, snapshot, resume):
    wandb.init(project='Dog-VC')
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for tag in ['gen', 'dis']:
        checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume, tag))
        if os.path.exists(checkpointpath):
            checkpoint = torch.load(checkpointpath, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
            print('{} loaded successfully.'.format(checkpointpath))

    w_adv = config['w_adv']
    w_grad = config['w_grad']
    w_cls = config['w_cls']
    w_cyc = config['w_cyc']
    w_rec = config['w_rec']
    gradient_clip = config['gradient_clip']
    wav_test_dir = f"{config['test_dir_path']}/wavs"
    val_each_step = config['val_step']

    vocoder = torch.jit.load(config["vocoder_path"], map_location="cpu")

    print("===================================Training Started===================================")
    n_iter = 0
    for epoch in range(resume + 1, epochs + 1):
        b = 0
        for batch in train_loader:
            models['gen'].train()
            models['dis'].train()

            ids, human_mels, dog_mels, audio_sizes = batch[0]
            bs = len(ids)
            ids = ids.to(device)
            human_mels = human_mels.to(device).permute(0, 2, 1)
            dog_mels = dog_mels.to(device).permute(0, 2, 1)
            audio_sizes = audio_sizes.to(device)
            mel_masks = get_mask_from_lengths(audio_sizes, device).to(device)

            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0

            hum = torch.Tensor([0]).long().to(device)
            dog = torch.Tensor([1]).long().to(device)
            AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = models['stargan'].calc_gen_loss(human_mels, dog_mels, hum, dog,
                                                                                     mel_masks)
            gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)

            models['gen'].zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(models['gen'].parameters(), gradient_clip)
            optimizers['gen'].step()

            AdvLoss_d, GradLoss_d, ClsLoss_d = models['stargan'].calc_dis_loss(human_mels, dog_mels, hum, dog,
                                                                               mel_masks)
            dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

            models['dis'].zero_grad()
            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(models['dis'].parameters(), gradient_clip)
            optimizers['dis'].step()

            gen_loss_mean += gen_loss.item()
            dis_loss_mean += dis_loss.item()
            advloss_d_mean += AdvLoss_d.item()
            gradloss_d_mean += GradLoss_d.item()
            advloss_g_mean += AdvLoss_g.item()
            clsloss_d_mean += ClsLoss_d.item()
            clsloss_g_mean += ClsLoss_g.item()
            cycloss_mean += CycLoss.item()
            recloss_mean += RecLoss.item()

            n_iter += 1
            b += 1

        wandb.log({"train_loss/gen_loss": gen_loss_mean / b})
        wandb.log({"train_loss/dis_loss": dis_loss_mean / b})
        wandb.log({"train_loss/advloss_d": advloss_d_mean / b})
        wandb.log({"train_loss/gradloss_d": gradloss_d_mean / b})
        wandb.log({"train_loss/advloss_g": advloss_g_mean / b})
        wandb.log({"train_loss/clsloss_d": clsloss_d_mean / b})
        wandb.log({"train_loss/clsloss_g": clsloss_g_mean / b})
        wandb.log({"train_loss/cycloss": cycloss_mean / b})
        wandb.log({"train_loss/recloss": recloss_mean / b})

            # print(f"Epoch: {epoch} | Batch: {b} | Gen Loss: {gen_loss_mean}")
        if epoch % val_each_step == 0 or epoch == 1:
            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0

            models['gen'].eval()
            models['dis'].eval()
            with torch.no_grad():
                b = 1
                for batch in test_loader:
                    ids, human_mels, dog_mels, audio_sizes = batch[0]
                    mel_masks = get_mask_from_lengths(audio_sizes, device).to(device)
                    bs = len(ids)
                    ids = ids.to(device)
                    human_mels = human_mels.to(device).permute(0, 2, 1)
                    dog_mels = dog_mels.to(device).permute(0, 2, 1)
                    audio_sizes = audio_sizes.to(device)
                    hum = torch.Tensor([0]).long().to(device)
                    dog = torch.Tensor([1]).long().to(device)
                    AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = models['stargan'].calc_gen_loss(human_mels, dog_mels, hum,
                                                                                             dog, mel_masks)
                    val_gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)
                    # AdvLoss_d, GradLoss_d, ClsLoss_d = models['stargan'].calc_dis_loss(human_mels, dog_mels, hum, dog)
                    # val_dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

                    gen_loss_mean += val_gen_loss.item()
                    # dis_loss_mean += val_dis_loss.item()
                    # advloss_d_mean += AdvLoss_d.item()
                    # gradloss_d_mean += GradLoss_d.item()
                    advloss_g_mean += AdvLoss_g.item()
                    # clsloss_d_mean += ClsLoss_d.item()
                    clsloss_g_mean += ClsLoss_g.item()
                    cycloss_mean += CycLoss.item()
                    recloss_mean += RecLoss.item()

                    generated_h2d = models['stargan'](human_mels, hum, dog)
                    generated_d2h = models['stargan'](dog_mels, dog, hum)

                    # print(generated_h2d.shape)
                    # print(audio_sizes)

                    if epoch == 1:
                        for i in ids:
                            original_human_audio, sr = torchaudio.load(f"{wav_test_dir}/0_{i}.wav")
                            original_dog_audio, sr = torchaudio.load(f"{wav_test_dir}/1_{i}.wav")
                            original_human_audio = original_human_audio[0, :]
                            original_dog_audio = original_dog_audio[0, :]
                            # print(original_dog_audio.shape, original_human_audio.shape)
                            wandb.log(
                                {f"{i}/human": wandb.Audio(original_human_audio.detach().numpy(),
                                                           caption=f"{i}/human", sample_rate=sr)})

                            wandb.log(
                                {f"{i}/dog": wandb.Audio(original_dog_audio.detach().numpy(),
                                                         caption=f"{i}/dog", sample_rate=sr)})

                    for j, i in enumerate(ids):
                        h2d = generated_h2d[j, :, :audio_sizes[j]].unsqueeze(0)
                        wav_prediction = vocoder(h2d.detach().cpu())[0].squeeze(0).detach().cpu().numpy()
                        wandb.log({f"{i}/h2d": wandb.Audio(wav_prediction, caption=f"{i}/h2d", sample_rate=22050)})

                        d2h = generated_d2h[j, :, :audio_sizes[j]].unsqueeze(0)
                        wav_prediction = vocoder(d2h.detach().cpu())[0].squeeze(0).detach().cpu().numpy()
                        wandb.log({f"{i}/d2h": wandb.Audio(wav_prediction, caption=f"{i}/d2h", sample_rate=22050)})

                wandb.log({"val_loss/gen_loss": gen_loss_mean / b})
                # wandb.log({"val_loss/dis_loss": dis_loss_mean / b})
                # wandb.log({"val_loss/advloss_d": advloss_d_mean / b})
                # wandb.log({"val_loss/gradloss_d": gradloss_d_mean / b})
                wandb.log({"val_loss/advloss_g": advloss_g_mean / b})
                # wandb.log({"val_loss/clsloss_d": clsloss_d_mean / b})
                wandb.log({"val_loss/clsloss_g": clsloss_g_mean / b})
                wandb.log({"val_loss/cycloss": cycloss_mean / b})
                wandb.log({"val_loss/recloss": recloss_mean / b})

        if epoch % snapshot == 0:
            for tag in ['gen', 'dis']:
                print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                           os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")


def main():
    parser = argparse.ArgumentParser(description='StarGAN-VC')
    parser.add_argument('--gpu', '-g', type=int, default=5, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--vocoder_path', '-v', type=str,
                        default="/root/storage/dasha/saved_models/hifi_torchscript/rus_1_7M_hifi.pt",
                        help='Vocoder Path')
    parser.add_argument('-train_ddir', '--train_ddir', type=str,
                        default='/root/storage/dasha/data/dog_dataset/train_human_paired',
                        help='root data folder that contains the normalized features')
    parser.add_argument('-test_ddir', '--test_ddir', type=str,
                        default='/root/storage/dasha/data/dog_dataset/test_human_paired',
                        help='root data folder that contains the normalized features')
    parser.add_argument('--epochs', '-epoch', default=2000, type=int, help='number of epochs to learn')
    parser.add_argument('--val_step', '-val step', default=50, type=int, help='epoch after which run validation')
    parser.add_argument('--snapshot', '-snap', default=2000, type=int, help='snapshot interval')
    parser.add_argument('--batch_size', '-batch', type=int, default=32, help='Batch size')
    parser.add_argument('--num_mels', '-nm', type=int, default=80, help='number of mel channels')
    parser.add_argument('--arch_type', '-arc', default='conv', type=str, help='generator architecture type (conv or rnn)')
    parser.add_argument('--loss_type', '-los', default='wgan', type=str, help='type of adversarial loss (cgan, wgan, or lsgan)')
    parser.add_argument('--zdim', '-zd', type=int, default=16, help='dimension of bottleneck layer in generator')
    parser.add_argument('--hdim', '-hd', type=int, default=64, help='dimension of middle layers in generator')
    parser.add_argument('--mdim', '-md', type=int, default=32, help='dimension of middle layers in discriminator')
    parser.add_argument('--sdim', '-sd', type=int, default=16, help='dimension of speaker embedding')
    parser.add_argument('--lrate_g', '-lrg', default='0.0005', type=float, help='learning rate for G')
    parser.add_argument('--lrate_d', '-lrd', default='5e-6', type=float, help='learning rate for D/C')
    parser.add_argument('--gradient_clip', '-gclip', default='1.0', type=float, help='gradient clip')
    parser.add_argument('--w_adv', '-wa', default='1.0', type=float, help='Weight on adversarial loss')
    parser.add_argument('--w_grad', '-wg', default='1.0', type=float, help='Weight on gradient penalty loss')
    parser.add_argument('--w_cls', '-wcl', default='1.0', type=float, help='Weight on classification loss')
    parser.add_argument('--w_cyc', '-wcy', default='1.0', type=float, help='Weight on cycle consistency loss')
    parser.add_argument('--w_rec', '-wre', default='1.0', type=float, help='Weight on reconstruction loss')
    parser.add_argument('--normtype', '-norm', default='IN', type=str, help='normalization type: LN, BN and IN')
    parser.add_argument('--src_conditioning', '-srccon', default=0, type=int, help='w or w/o source conditioning')
    parser.add_argument('--resume', '-res', type=int, default=0, help='Checkpoint to resume training')
    parser.add_argument('--model_rootdir', '-mdir', type=str, default='./model/arctic/', help='model file directory')
    parser.add_argument('--log_dir', '-ldir', type=str, default='./logs/arctic/', help='log file directory')
    parser.add_argument('--experiment_name', '-exp', default='experiment1', type=str, help='experiment name')
    args = parser.parse_args()

    # Set up GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = 'cuda'
    else:
        device = 'cpu'

    # Configuration for StarGAN
    vocoder_path = args.vocoder_path
    val_step = args.val_step
    train_dir_path = args.train_ddir
    test_dir_path = args.test_ddir
    num_mels = args.num_mels
    arch_type = args.arch_type
    loss_type = args.loss_type
    zdim = args.zdim
    hdim = args.hdim
    mdim = args.mdim
    sdim = args.sdim
    w_adv = args.w_adv
    w_grad = args.w_grad
    w_cls = args.w_cls
    w_cyc = args.w_cyc
    w_rec = args.w_rec
    lrate_g = args.lrate_g
    lrate_d = args.lrate_d
    gradient_clip = args.gradient_clip
    epochs = args.epochs
    batch_size = args.batch_size
    snapshot = args.snapshot
    resume = args.resume
    normtype = args.normtype
    src_conditioning = bool(args.src_conditioning)

    train_loader = get_dataloader(data_dir_path=train_dir_path, batch_size=batch_size, drop_last=True,
                                  shuffle=True, sort=True, num_workers=16)
    test_loader = get_dataloader(data_dir_path=test_dir_path, batch_size=batch_size, drop_last=True,
                                 shuffle=False, sort=False, num_workers=16)

    n_spk = 2
    spk_list = [0, 1]
    model_config = {
        'vocoder_path': vocoder_path,
        'val_step': val_step,
        'train_dir_path': train_dir_path,
        'test_dir_path': test_dir_path,
        'num_mels': num_mels,
        'arch_type': arch_type,
        'loss_type': loss_type,
        'zdim': zdim,
        'hdim': hdim,
        'mdim': mdim,
        'sdim': sdim,
        'w_adv': w_adv,
        'w_grad': w_grad,
        'w_cls': w_cls,
        'w_cyc': w_cyc,
        'w_rec': w_rec,
        'lrate_g': lrate_g,
        'lrate_d': lrate_d,
        'gradient_clip': gradient_clip,
        'normtype': normtype,
        'epochs': epochs,
        'n_spk': n_spk,
        'spk_list': spk_list,
        'src_conditioning': src_conditioning
    }
    model_dir = os.path.join(args.model_rootdir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, args.experiment_name, 'train_{}.log'.format(args.experiment_name))

    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(model_config, outfile, indent=4)

    if arch_type == 'conv':
        gen = net.Generator1(num_mels, n_spk, zdim, hdim, sdim, normtype, src_conditioning)
    elif arch_type == 'rnn':
        net.Generator2(num_mels, n_spk, zdim, hdim, sdim, src_conditioning=src_conditioning)
    dis = net.Discriminator1(num_mels, n_spk, mdim, normtype)
    models = {'gen': gen, 'dis': dis}
    models['stargan'] = net.StarGAN(models['gen'], models['dis'], n_spk, loss_type)

    optimizers = {
        'gen': optim.Adam(models['gen'].parameters(), lr=lrate_g, betas=(0.9, 0.999)),
        'dis': optim.Adam(models['dis'].parameters(), lr=lrate_d, betas=(0.5, 0.999))
    }

    for tag in ['gen', 'dis']:
        models[tag].to(device).train(mode=True)

    Train(models, epochs, train_loader, test_loader, optimizers, device, model_dir, log_path, model_config, snapshot,
          resume)


if __name__ == '__main__':
    main()

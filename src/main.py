#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 21 20:42 2020

@author: phongdk
"""
import sys
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks import *
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")

N_EPOCHS = 25
BATCH_SIZE = 16
w = 0.55
lr = 0.01

block_images = ['f72fb0a92acc205c8781cce71039a638.jpg', 'ad9d85c12b90b350edd61cb251d55d4f.jpg',
                '4d9ea01673bda390208bfacd7e4e6458.jpg', '9e4906af770b8499171c40e9e2f96a23.jpg',
                '082281a06e8e746aa3fed68d0dba5e56.jpg', '80d72a2c68e7a736e857798630dde127.jpg',
                'ea41dee44e4bf712022f31965b2d9017.jpg', '210e2f0254319207b6056388e16f8743.jpg',
                '41023cee36923b8a4f71c85e0df68dde.jpg', '152d1fa5487bd87565a02c81d4d1506d.jpg',
                '543638dea9bba62f37823351e21227ac.jpg', 'efb5f2e8551d8f1117b23c5ce9208a5f.jpg',
                '5aa5671e9e19101dd9d350cd8c746ba0.jpg', '453c2532e496dc29620072bc35ce37fa.jpg',
                '5909ed365d18b580fb04d2be6b0567a9.jpg', '4eae1ee162653a6bb07c05cae44cca9e.jpg',
                'ced339545a1ce65eca0f712d5691cbfa.jpg', 'e3d2bc74de8864a994c895cf28d42831.jpg',
                '35bd9c40c499299da0f11c365de25085.jpg', '96aadd8b9a37f54bd099f676aafe2353.jpg',
                'abd0ad01966eaef8217ba39812b84dba.jpg', 'fb88714e8a6bc33e4f3ce79971694fea.jpg',
                'e86710a18935bfe0313acf4e976dde5c.jpg', 'b417630b2ffc164e2c998bf49e7810ab.jpg',
                '24997a6a9db5bceaae126055ffa0341f.jpg', '9af0bab45f6850c6deba9b49ac5bbe8f.jpg',
                'fcd38658683277618d6670270569bea7.jpg', '3cf53cb46896a73cb180a0501636b607.jpg',
                'c5950c0913d13d3bf30bd3c5778df7f0.jpg', '088e57cb962dfc20a637938357459aa0.jpg',
                '3e5e0a5cb785a22ed8ce3774b21e257e.jpg', 'f4969433087a9c560dd4a04993e6f25f.jpg',
                '6a194c1f63119c9e295c2d3a817b4ca6.jpg', '7c20f48078c35b85435d517c862e4beb.jpg',
                '8ff1d9eae0c27b8e61dd755a9c9d41dd.jpg', '32b8c8586089022e1d8c8bd788e3e5e5.jpg',
                'bddde6e4f08c5cee98211bd243f35083.jpg', '06658ddee340c1e791f081cc4552824e.jpg',
                '69c4fb9827a75d9f99f21b89c5497cb7.jpg', 'b305f1d8a0788888a3f44b14f878350d.jpg',
                '2d72d6f4249fbdde0524803c9ef40965.jpg', '59eb98d9977e60e5b73f64b9efac8d4e.jpg',
                'b6a1b25263469f2c3a22d39eb948374e.jpg', 'dcb894896dd08cbdfc6254cd8aa631e8.jpg',
                'bca3826f9bf3f94ed041ac9eb6419295.jpg', 'e0f51d6f970154e7be697f72791564e0.jpg',
                '177d8071710ba0043d6b342340b6547f.jpg', '1021ca9e7090fad3f15457e9a877a260.jpg',
                'e9e30d25389300ca3d09c6d82e18ac18.jpg', '71d2aea099e4aace5b0015958ace1d3c.jpg',
                '15d502b8f2c1192846e933e7340931b0.jpg', 'aa6c4361e6ad764d4120a4d0056a4da3.jpg',
                'eb9be544776b2bd400f6ae41593b5a2c.jpg', '236ce5245d6e78adc38907ce8ddbabe8.jpg',
                'af7dd1cd3be71f0d64abc48ee8182d98.jpg', '657d6037df07435e9ae2a30e240a3898.jpg',
                '512134ed41bca1d8aed32cb3e38b44ab.jpg', '4d5dff3f1fc8a9c41b9e61a94b4e9623.jpg',
                '0ac37f5751a9207a8583a66582398544.jpg', '8d434088d20d1d87f5ba5a31df56b7d6.jpg',
                '944c2c4f5116526d5d29f2e63ba2f371.jpg', 'ffdbd4c11fb38e7d200a94819f9e5629.jpg',
                'd20d349139af3a1c186a9edb098a918f.jpg', '8985fa5007ddbea4e0b4a5db9050c5e4.jpg',
                'a7d332ec5361d05938f059bdc4e33124.jpg', 'ca8e0f74515b41a46ec0c755b064d3ee.jpg',
                '05d5badd3cb44235c20cd12570eb1e96.jpg', '3a933b3d8e94d249f0b0ae141bb18b35.jpg',
                'c730841deeb2aae033c9c97325a416bd.jpg', 'dcae1c2a24fa9d590508e2aea61f8265.jpg',
                'cae6f0ba3bb19b8bdabdc59c608ac581.jpg', '1318ca4450aa2fba30ffc10e41489c68.jpg',
                'a71eb495583638591ac66acfa0fb161a.jpg', '0f29bcd1e497aef3047313dc70f38615.jpg',
                '05bf79da83141e5898985cd40b6680f3.jpg', 'aac6d61074e67b6f62fc507e52f72164.jpg',
                '4cc1b14763fc7b32f8c945a80c2694e9.jpg', '486de41a9ad0df3f5963db3e0b3c87cf.jpg',
                '8dc1a7a39bb84d07a9e793bfea562e76.jpg', 'a6872a34183290faf8f9d2a625e7db9e.jpg',
                '6f58ef6326147e68f35cf89e18acb02f.jpg', '2e8b9643fe1bd906130424ef98fd7d5c.jpg',
                '0bccaa942eef0c7b721a4b1816becebd.jpg', '2d86c356cdea22ed0650d7a765678229.jpg',
                'dee9a351fec5a82322702f9653aca636.jpg', '99dabaf885a883d24a651d1a5b3617f7.jpg',
                '3f48d825ad8c576ffdf5aed066310ab0.jpg', '6d7db3225ae2fba229023279662e9a7d.jpg',
                '9e6ea514301eddc9af080e45b14747e6.jpg', '98b3f88cc867a13fd56086b965376f06.jpg',
                'd0afaa0d304fe2818f8c16a0f8529aff.jpg', '7f6752a762da132f0ef793fb8016146f.jpg',
                '0c355c122131fe279471b9c2cea8a2cd.jpg', 'e28b7ddd7cd4406380abb9f8c656f97a.jpg',
                '33bc9a2646dab6da5326aa50f9664341.jpg', 'dbbdb01559cd322d9881c5e924561684.jpg',
                'f4b04f68843e9615c04f0c7424bca90c.jpg', '25cee358dbbf65271a31ca54c964a0f7.jpg',
                'b887c039551ff0d66550dd334fd8320f.jpg', '2f57c979932143a424e939fb1f533a9b.jpg',
                'e8dd8f7cd80787d7bf74abf4186b6dbb.jpg', 'ae5872d750860665f213706b356d61f1.jpg',
                '52cf44d8babd7f1b8dead3373cd102ca.jpg', 'f7c1c107c5861a55df9d6efa030b9b8f.jpg',
                '46cf80986d0879d1952070ad1fdbf01b.jpg', 'b59a3b936c5fe6241157b1923d2e2f82.jpg',
                '2273416d010e3e928f31bbdacd970d19.jpg', 'b97b87e04907f39bff3de893a45296fd.jpg',
                '6de190f00bec682b281c1f74a51eb533.jpg', '96f394a22ffc740ce01269fb4cfff7c0.jpg',
                '86c672f0549196413d4d188b8f8c8fdf.jpg', 'a95d37805d8cdec7acafbf7ead278808.jpg',
                '4e3b0079d28117f1e2acb86d46b9a1dd.jpg', '1619707a95815f8af727701f8cab6670.jpg',
                'cbac1dd771276df0ee1b8c9bee77cff8.jpg', '84aab8477eccb5c5107f0b2129e0725c.jpg',
                '1f681f0dfea883111c747284d8cec4d0.jpg', 'c244d292952833271072edb882823f48.jpg',
                'e1a320a8aacea1b7bb1f5e0f9447931f.jpg', '80dacf5798f7266594471faf07e744a4.jpg',
                '050af5138374b6a3331286fa812a4d4b.jpg', '6e6616b642d2db4cf1dcd7356261d6db.jpg',
                '42cfa67952c0d9d16f21d7c10353788c.jpg', '59b4814876fe3c682a10932ff90806aa.jpg',
                'ffdca50c0598b74281341481f49509ba.jpg', '197a7e82333f7f2632a8a5659216e082.jpg',
                '780037c28387e1de3fce270779230c20.jpg', 'ce3eb1d873dd9c2ca0242da16c369b98.jpg',
                '8cab91298cf38f4588cef456dcd143cd.jpg', '574fee5d3bff42572d8884851bf0f9ef.jpg',
                '0ba38dbfb0d8d7e127fc78c0d05d3f80.jpg', 'd8b987dde348e93cbeb0d0b865349adf.jpg',
                '6f5a068be16054c14ce69ac50a0c0255.jpg', '9cbe9c2e139d67e259f664768bdb4941.jpg',
                '32e4abcce96678dcbf9de57d4edd1603.jpg', 'a58a0e467364522d8968d5eac4b07fbb.jpg',
                '411faf9d3eac2232261192fb63135586.jpg', '27161fd8eca8f566d27c166b23b16471.jpg',
                '86c4553ed465d09cad0ec83cb59deed8.jpg', 'c1613b493e7c8023628b83dfbe9a9cd6.jpg',
                '7c5c15f0390ef5aeb282e68c741a9edf.jpg', 'bc9ee7e7cfb1ecc2cb63df0946d6ada1.jpg',
                'c20fa00e2543d3139578739e9f10fae9.jpg', '98e8e05ead435d22628a6e0e3be98d85.jpg',
                '767fa4d1c451e1cd605be9c146db9c12.jpg', '122c897c122b180622dafb5eaa3b85ad.jpg',
                'cb99d76b8c8fef751c4be083c6c13102.jpg', 'e1a1fcd2de7a666a801116334e16f4c8.jpg',
                '786dd61218a8fa62176cdea4a898b3da.jpg', '23a1a741cf31456ecd86dbc0340d7908.jpg',
                'b5e41c0a4cdeb3cd02a9d7473bf86a3f.jpg', '42c53dec3636f097150b9f7e88f02d95.jpg',
                '1a81e2bfa6600c7ee91df87b909881aa.jpg', '80c62aa2c8d88493f8ddd737e7eae5bb.jpg',
                '8f03038a6fd4b01f9d29c929a4417151.jpg', 'b13a9a3abfcdf15dc7bea60f71a956cd.jpg',
                '271b362c0b283f8d671a342d938a8443.jpg', '4ecb99b0f9157c23cd752a47a413178c.jpg',
                '8a77015aae6246f5a5baaa1c330ba00a.jpg', '00a6280fee57e8f8e9338d2c4d0a9ac8.jpg',
                'acb6cdc9964c9a56de5524fb8e90f377.jpg', '0aa9fc5a25901cf3d03a27e60c002e14.jpg',
                'e071515f94cae44432bc8f7b96202602.jpg', 'dfb0d1361a542ae0fccd6536547738de.jpg',
                '194f75cbdd45d2c4a6b56224c02e9d6c.jpg', 'e55b6167dbe04b49a8415b7beaa2baf7.jpg',
                '41243bdd1557bca78b89a2f95b0c62a9.jpg', '5e8133bd897df7dc6fb51e577e2f6836.jpg',
                '6dfd2fea861381bbeaa94d938f052711.jpg', 'd877c237d29a64906ba34190d00a682e.jpg',
                '5adc575dc763b5753af72adc11543dba.jpg', '9ca82721e13a8be7826d26b1a0005a85.jpg',
                '18773414099779569cab7e9e7f1ae773.jpg', '16206a46a9ff0c19f84995d28ef50ec7.jpg',
                'd4bc7e1e98e4522cbf0265cf16dc3fac.jpg', '2b726b52f4a950e1184046dade5dd57f.jpg',
                '8eb87525e94b48c83221b51b67378e0b.jpg', 'b01cda4f977fdc1aa9c9216a482e3433.jpg',
                '0f398fc20ce4902bfed5c84e37bc0a65.jpg', '5556c227eeb13dd8898d0654f293082e.jpg',
                'a30799c9094691158313cb02e152a34d.jpg', '3dac4d5c47bd6262b94ab012a5527155.jpg',
                '0de66444ddd50e766700116b07083aa3.jpg', '2511dd7cd66d2c3f924a031b6a96d853.jpg',
                '52fc98dbd539113c15879881ab79fcde.jpg', '80604ec0eeb4353f53e05572b5b4765e.jpg',
                '7c33bff03ad5484f5a00d3bd2cdbb77f.jpg', 'a35a1d09432d1ff4ce1a311c46851c49.jpg',
                'd84b12353e3347f46a7f4178f13567e2.jpg', 'ee7a8e537e2b19ef86816fca4d65579c.jpg',
                '93b77969da60de96a14488635b6f56ef.jpg', 'f14996d8066af9c0479c9036a09dd1ee.jpg',
                '535a3b9b9a6aaa82b8be9699b227082d.jpg', '729b626083142df78571dbfa09ce7174.jpg',
                '715d94ab9d8135b7a9bf17f704895358.jpg', 'cf75e0ddc421a017e68e94721ec3b853.jpg',
                '7f3495a0074a0af2438f5e9efb063494.jpg', '2814ce379a74707516da91d3d13ac136.jpg',
                '3e5a3d0cf18e2905388fed1e9d554cdd.jpg', '5feb1ef8031c4f35c9af9e6bb0e9d22f.jpg']


def load_data(train_path, test_path):
    print('---------------Load data---------------')
    test_data = ImageList.from_folder(test_path)

    tfms = get_transforms()
    train_data = (ImageList.from_folder(train_path)
                  .filter_by_func(lambda fname: Path(fname).name not in block_images)
                  .split_by_rand_pct(valid_pct=0.15, seed=42)
                  .label_from_folder()
                  .transform(tfms, size=(224, 224))
                  .databunch()
                  .normalize(imagenet_stats))
    return train_data, test_data


def train_model(model, train_data, lr=lr, batch_size=BATCH_SIZE):
    print('--------------Train model -----------------')
    learn = cnn_learner(train_data,
                        model,
                        metrics=accuracy,
                        model_dir=os.getcwd()).mixup()

    learn.data.batch_size = batch_size
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6 ** i))) for i in range(N_EPOCHS)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.callbacks.append(SaveModelCallback(learn, every='improvement', monitor='accuracy', name='best'))
    learn.fit_one_cycle(N_EPOCHS)
    return learn


def predict_test(learn, il):
    print('--------------- Data Prediction---------------')
    predicts = []
    for image in tqdm(il):
        predicts.append(to_np(learn.predict(image)[-1]))
    predicts = np.stack(predicts)
    return predicts


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]  # '/home/phongdk/shopee/'
    train_path = f'{DATA_DIR}/train/train/'
    test_path = f'{DATA_DIR}/test/test/'

    train_data, test_data = load_data(train_path, test_path)
    learn = train_model(models.resnet152, train_data, lr=lr)
    learn.save('resnet152_bi.model', return_path=True)
    # learn.load('resnet152_bi.model')
    predicts_1 = predict_test(learn, test_data)
    del learn
    gc.collect()

    learn = train_model(models.densenet201, train_data, lr=lr)
    learn.save('densenet201_bi.model', return_path=True)
    predicts_2 = predict_test(learn, test_data)
    # learn.load('densenet201_bi.model')
    predicts = np.argmax(predicts_1 * w + predicts_2 * (1 - w), axis=1)
    print(predicts)
    sub = pd.DataFrame({'filename': test_data.items, 'category': predicts})
    sub['filename'] = sub['filename'].apply(lambda x: str(x).split('/')[-1])
    submission = pd.read_csv(f"{DATA_DIR}/test.csv")
    del submission['category']
    submission = submission.merge(sub, how='left', on='filename')
    submission['category'] = submission['category'].astype(str).apply(lambda x: x.zfill(2))
    submission[['filename', 'category']].to_csv('submission.csv', index=False)

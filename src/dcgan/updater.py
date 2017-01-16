import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, gen, dis, *args, **kwargs):
        self.gen = gen
        self.dis = dis
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = y_fake.data.shape[0]
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = y_fake.data.shape[0]
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        batchsize = len(batch)

        y_real = self.dis(x_real, test=False)

        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = self.gen(z, test=False)
        y_fake = self.dis(x_fake, test=False)

        dis_optimizer.update(self.loss_dis, self.dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, self.gen, y_fake)

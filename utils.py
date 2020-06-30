from torch.utils.tensorboard.summary import hparams

def better_hparams(writer, hparam_dict=None, metric_dict=None):
    """Add a set of hyperparameters to be compared in TensorBoard.
    Args:
        hparam_dict (dictionary): Each key-value pair in the dictionary is the
          name of the hyper parameter and it's corresponding value.
        metric_dict (dictionary): Each key-value pair in the dictionary is the
          name of the metric and it's corresponding value. Note that the key used
          here should be unique in the tensorboard record. Otherwise the value
          you added by `add_scalar` will be displayed in hparam plugin. In most
          cases, this is unwanted.

        p.s. The value in the dictionary can be `int`, `float`, `bool`, `str`, or
        0-dim tensor
    Examples::
        from torch.utils.tensorboard import SummaryWriter
        with SummaryWriter() as w:
            for i in range(5):
                w.add_hparams({'lr': 0.1*i, 'bsize': i},
                              {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
    Expected result:
    .. image:: _static/img/tensorboard/add_hparam.png
       :scale: 50 %
    """
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    # writer.file_writer.add_summary(sei)
    # for k, v in metric_dict.items():
    #     writer.add_scalar(k, v)
    # with SummaryWriter(log_dir=os.path.join(self.file_writer.get_logdir(), str(time.time()))) as w_hp:
    #     w_hp.file_writer.add_summary(exp)
    #     w_hp.file_writer.add_summary(ssi)
    #     w_hp.file_writer.add_summary(sei)
    #     for k, v in metric_dict.items():
    #         w_hp.add_scalar(k, v)

    return sei

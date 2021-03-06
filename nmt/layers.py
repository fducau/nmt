import theano
import theano.tensor as tensor
import numpy
from utils import *

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'ff_nb': ('param_init_fflayer_nb', 'fflayer_nb'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond_simple': ('param_init_gru_cond_simple', 'gru_cond_simple_layer'),
          'gru_hiero': ('param_init_gru_hiero', 'gru_hiero_layer'),
          'rnn': ('param_init_rnn', 'rnn_layer'),
          'rnn_cond': ('param_init_rnn_cond', 'rnn_cond_layer'),
          'rnn_hiero': ('param_init_rnn_hiero', 'rnn_hiero_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


# dropout
def dropout_layer(state_before, use_noise, trng, keep_p=0.5):
    proj = tensor.switch(use_noise,
                         state_before * trng.binomial(state_before.shape,
                                                      p=keep_p, n=1,
                                                      dtype=state_before.dtype),
                         state_before * keep_p)
    return proj


# LINEAR FORWARD LAYERS
########################

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[prefix_append(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[prefix_append(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below,
                                  tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')])


# feedforward layer with no bias: affine transformation + point-wise nonlinearity
def param_init_fflayer_nb(options, params, prefix='ff_nb', nin=None, nout=None, ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[prefix_append(prefix,'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)

    return params


def fflayer_nb(tparams, state_below, options, prefix='ff_nb', activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]))


# RNN LAYERS
###############
# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)],
                              axis=1)

        params[prefix_append(prefix, 'W')] = W
        params[prefix_append(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prefix_append(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[prefix_append(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prefix_append(prefix, 'Ux')] = Ux
    params[prefix_append(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[prefix_append(prefix,'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wx')]) + tparams[prefix_append(prefix, 'bx')]
    U = tparams[prefix_append(prefix, 'U')]
    Ux = tparams[prefix_append(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h  #, r, u, preact, preactx

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(0., n_samples, dim)],
                                non_sequences=[tparams[prefix_append(prefix, 'U')],
                                               tparams[prefix_append(prefix, 'Ux')]],
                                name=prefix_append(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond', nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru_nonlin(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wc = norm_weight(dimctx, dim * 2)
    params[prefix_append(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[prefix_append(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[prefix_append(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prefix_append(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prefix_append(prefix, 'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx, 1)
    params[prefix_append(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prefix_append(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix, 'Wc_att')]) + tparams[prefix_append(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wx')]) + tparams[prefix_append(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')]
    #state_belowc = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wi_att')])
    #import ipdb; ipdb.set_trace()

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):

        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)   # (4)
        u1 = _slice(preact1, 1, dim)   # (5)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)  # (3)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_     # (2)

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        # pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])     # (8)
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)     # (7)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)     # (6) current context

        preact2 = tensor.dot(h1, U_nl) + b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T #, pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[prefix_append(prefix, 'U')],
                   tparams[prefix_append(prefix, 'Wc')],
                   tparams[prefix_append(prefix, 'W_comb_att')],
                   tparams[prefix_append(prefix, 'U_att')],
                   tparams[prefix_append(prefix, 'c_tt')],
                   tparams[prefix_append(prefix, 'Ux')],
                   tparams[prefix_append(prefix, 'Wcx')],
                   tparams[prefix_append(prefix, 'U_nl')],
                   tparams[prefix_append(prefix, 'Ux_nl')],
                   tparams[prefix_append(prefix, 'b_nl')],
                   tparams[prefix_append(prefix, 'bx_nl')]]

    if one_step:
        rval = _step( * (seqs + [init_state, None, None, pctx_, context] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples, context.shape[2]),
                                                  tensor.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_, context] + shared_vars,
                                    name=prefix_append(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def param_init_gru_nonlin(options, params, prefix='gru', nin=None, dim=None, hiero=False):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[prefix_append(prefix,'W')] = W
        params[prefix_append(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prefix_append(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[prefix_append(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prefix_append(prefix,'Ux')] = Ux
    params[prefix_append(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    
    U_nl = numpy.concatenate([ortho_weight(dim),
                              ortho_weight(dim)], axis=1)
    params[prefix_append(prefix,'U_nl')] = U_nl
    params[prefix_append(prefix,'b_nl')] = numpy.zeros((2 * dim,)).astype('float32')

    Ux_nl = ortho_weight(dim)
    params[prefix_append(prefix,'Ux_nl')] = Ux_nl
    params[prefix_append(prefix,'bx_nl')] = numpy.zeros((dim,)).astype('float32')
    
    return params


# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None, hiero=False):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    if not hiero:
        W = numpy.concatenate([norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim),
                               norm_weight(nin,dim)], axis=1)
        params[prefix_append(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prefix_append(prefix,'U')] = U
    params[prefix_append(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[prefix_append(prefix,'U')].shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[prefix_append(prefix, 'U')])
        preact += x_
        preact += tparams[prefix_append(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c, i, f, o, preact

    state_below = tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')]

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info = [tensor.alloc(0., n_samples, dim),
                                                tensor.alloc(0., n_samples, dim),
                                                None, None, None, None],
                                name=prefix_append(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    return rval


# Conditional GRU layer without Attention
def param_init_gru_cond_simple(options, params, prefix='gru_cond', nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']


    params = param_init_gru(options, params, prefix, nin=nin, dim=dim)


    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[prefix_append(prefix,'Wc')] = Wc

    Wcx = norm_weight(dimctx,dim)
    params[prefix_append(prefix,'Wcx')] = Wcx

    return params


def gru_cond_simple_layer(tparams, state_below, options, prefix='gru', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prefix_append(prefix, 'Ux')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 2, 'Context must be 2-d: #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wc')])
    pctxx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wcx')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wx')]) + tparams[prefix_append(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, pctx_, pctxx_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += pctx_
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += pctxx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[prefix_append(prefix, 'U')],
                   tparams[prefix_append(prefix, 'Ux')]] 

    if one_step:
        rval = _step(*(seqs+[init_state, pctx_, pctxx_]+shared_vars))
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=seqs,
                                    outputs_info=[init_state], 
                                    non_sequences=[pctx_,
                                                   pctxx_]+shared_vars,
                                    name=prefix_append(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond', nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_lstm(options, params, prefix, nin, dim)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*4)
    params[prefix_append(prefix,'Wc')] = Wc

    # attention: prev -> hidden
    Wi_att = norm_weight(nin,dimctx)
    params[prefix_append(prefix,'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prefix_append(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[prefix_append(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prefix_append(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[prefix_append(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix, 'c_tt')] = c_att

    return params


def lstm_cond_layer(tparams, state_below, options, prefix='lstm', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_memory, 'previous memory must be provided'
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prefix_append(prefix, 'U')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)
    # initial/previous memory 
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wc_att')]) + tparams[prefix_append(prefix,'b_att')]

    # projected x
    state_below = tensor.dot(state_below, tparams[prefix_append(prefix, 'W')]) + tparams[prefix_append(prefix, 'b')]
    state_belowc = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wi_att')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, xc_, h_, c_, ctx_, alpha_, pctx_):

        # attention
        pstate_ = tensor.dot(h_, tparams[prefix_append(prefix,'Wd_att')])
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, tparams[prefix_append(prefix,'U_att')])+tparams[prefix_append(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preact = tensor.dot(h_, tparams[prefix_append(prefix, 'U')])
        preact += x_
        preact += tensor.dot(ctx_, tparams[prefix_append(prefix, 'Wc')])

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c = tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:,None] * c + (1. - m_)[:,None] * c_

        h = o * tensor.tanh(c)
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h, c, ctx_, alpha.T, pstate_, preact, i, f, o

    if one_step:
        rval = _step(mask, state_below, state_belowc, init_state, init_memory, None, None, pctx_)
    else:
        rval, updates = theano.scan(_step, 
                                    sequences=[mask, state_below, state_belowc],
                                    outputs_info = [init_state, init_memory,
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0]),
                                                    None, None, None, 
                                                    None, None],
                                    non_sequences=[pctx_],
                                    name=prefix_append(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile)
    return rval


# Conditional RNN layer with Attention
def param_init_rnn_cond(options, params, prefix='rnn_cond', nin=None, dim=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_rnn(options, params, prefix, nin=nin, dim=dim)

    # context to LSTM
    Wcx = norm_weight(dimctx,dim)
    params[prefix_append(prefix,'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin,dimctx)
    params[prefix_append(prefix,'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prefix_append(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[prefix_append(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prefix_append(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[prefix_append(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix, 'c_tt')] = c_att

    return params


def rnn_cond_layer(tparams, state_below, options, prefix='rnn', mask=None, context=None, one_step=False, init_memory=None, init_state=None, context_mask=None, **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prefix_append(prefix, 'Ux')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wc_att')]) + tparams[prefix_append(prefix,'b_att')]
    pctx_ += tparams[prefix_append(prefix,'b_att')]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wx')]) + tparams[prefix_append(prefix, 'bx')]
    state_belowx += tparams[prefix_append(prefix, 'bx')]
    state_belowc = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wi_att')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, xx_, xc_, h_, ctx_, alpha_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wcx):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:, :, None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T  #, pstate_, preact, preactx, r, u

    if one_step:
        rval = _step(mask, state_belowx, state_belowc, init_state, None, None, 
                     pctx_, tparams[prefix_append(prefix,'Wd_att')],
                     tparams[prefix_append(prefix,'U_att')],
                     tparams[prefix_append(prefix, 'c_tt')],
                     tparams[prefix_append(prefix, 'Ux')],
                     tparams[prefix_append(prefix, 'Wcx')] )
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_belowx, state_belowc],
                                    outputs_info = [init_state, 
                                                    tensor.alloc(0., n_samples, context.shape[2]),
                                                    tensor.alloc(0., n_samples, context.shape[0])],
                                                    #None, None, None, 
                                                    #None, None],
                                    non_sequences=[pctx_,
                                                   tparams[prefix_append(prefix,'Wd_att')],
                                                   tparams[prefix_append(prefix,'U_att')],
                                                   tparams[prefix_append(prefix, 'c_tt')],
                                                   tparams[prefix_append(prefix, 'Ux')],
                                                   tparams[prefix_append(prefix, 'Wcx')]
                                                   ],
                                    name=prefix_append(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile)
    return rval


# RNN layer
def param_init_rnn(options, params, prefix='rnn', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    Wx = norm_weight(nin, dim)
    params[prefix_append(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prefix_append(prefix,'Ux')] = Ux
    params[prefix_append(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[prefix_append(prefix,'Ux')].shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_belowx = tensor.dot(state_below, tparams[prefix_append(prefix, 'Wx')]) + tparams[prefix_append(prefix, 'bx')]

    def _step(m_, xx_, h_, Ux):
        preactx = tensor.dot(h_, Ux)
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h#, r, u, preact, preactx

    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_belowx],
                                outputs_info = [tensor.alloc(0., n_samples, dim)],
                                                #None, None, None, None],
                                non_sequences=[tparams[prefix_append(prefix, 'Ux')]], 
                                name=prefix_append(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)
    rval = [rval]
    return rval


# Hierarchical GRU layer 
def param_init_gru_hiero(options, params, prefix='gru_hiero', nin=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    dim = dimctx

    params = param_init_gru(options, params, prefix, nin=nin, dim=dim, hiero=True)

    # context to LSTM
    Wc = norm_weight(dimctx,dim*2)
    params[prefix_append(prefix,'Wc')] = Wc

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prefix_append(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[prefix_append(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prefix_append(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[prefix_append(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix, 'c_tt')] = c_att

    # stop probability:
    W_st = norm_weight(dim, 1)
    params[prefix_append(prefix,'W_st')] = W_st
    b_st = -0. * numpy.ones((1,)).astype('float32')
    params[prefix_append(prefix,'b_st')] = b_st

    return params


def gru_hiero_layer(tparams, context, options, prefix='gru_hiero', context_mask=None, **kwargs):

    nsteps = context.shape[0]
    if context.ndim == 3:
        n_samples = context.shape[1]
    else:
        n_samples = 1

    # mask
    if context_mask is None:
        mask = tensor.alloc(1., context.shape[0], 1)
    else:
        mask = context_mask

    dim = tparams[prefix_append(prefix, 'W_st')].shape[0]

    # initial/previous state
    init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wc_att')]) + tparams[prefix_append(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step_slice(m_, h_, ctx_, alpha_, v_, pp_, cc_,
                    U, Wc, Wd_att, U_att, c_tt, Ux, Wx, bx, W_st, b_st):
        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pp_ + pstate_[None,:,:] 
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx = (cc_ * alpha[:,:,None]).sum(0) # current context

        preact = tensor.dot(h_, U)
        preact += tensor.dot(ctx, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx += tensor.dot(ctx, Wx)
        preactx += bx

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        # compute stopping probability
        ss = tensor.nnet.sigmoid(tensor.dot(h, W_st) + b_st)
        v_ = v_ * (1. - ss)[:,0][:,None]

        return h, ctx, alpha.T, v_[:,0] #, pstate_, preact, preactx, r, u

    _step = _step_slice

    rval, updates = theano.scan(_step, 
                                sequences=[mask],
                                outputs_info = [init_state, 
                                                tensor.alloc(0., n_samples, context.shape[2]),
                                                tensor.alloc(0., n_samples, context.shape[0]),
                                                tensor.alloc(1., n_samples)],
                                                #None, None, None, 
                                                #None, None],
                                non_sequences=[pctx_, 
                                               context,
                                               tparams[prefix_append(prefix, 'U')],
                                               tparams[prefix_append(prefix, 'Wc')],
                                               tparams[prefix_append(prefix,'Wd_att')], 
                                               tparams[prefix_append(prefix,'U_att')], 
                                               tparams[prefix_append(prefix, 'c_tt')], 
                                               tparams[prefix_append(prefix, 'Ux')], 
                                               tparams[prefix_append(prefix, 'Wx')],
                                               tparams[prefix_append(prefix, 'bx')], 
                                               tparams[prefix_append(prefix, 'W_st')], 
                                               tparams[prefix_append(prefix, 'b_st')]],
                                name=prefix_append(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)

    rval[0] = rval[0] * rval[3][:,:,None]
    return rval


# Hierarchical RNN layer 
def param_init_rnn_hiero(options, params, prefix='rnn_hiero', nin=None, dimctx=None):
    if nin is None:
        nin = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    dim = dimctx

    params = param_init_rnn(options, params, prefix, nin=nin, dim=dim)

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prefix_append(prefix,'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim,dimctx)
    params[prefix_append(prefix,'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prefix_append(prefix,'b_att')] = b_att

    # attention: 
    U_att = norm_weight(dimctx,1)
    params[prefix_append(prefix,'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix, 'c_tt')] = c_att

    # stop probability:
    W_st = norm_weight(dim, 1)
    params[prefix_append(prefix,'W_st')] = W_st
    b_st = numpy.zeros((1,)).astype('float32')
    params[prefix_append(prefix,'b_st')] = b_st

    return params


def rnn_hiero_layer(tparams, context, options, prefix='rnn_hiero', context_mask=None, **kwargs):

    nsteps = context.shape[0]
    if context.ndim == 3:
        n_samples = context.shape[1]
    else:
        n_samples = 1

    # mask
    if context_mask is None:
        mask = tensor.alloc(1., context.shape[0], 1)
    else:
        mask = context_mask

    dim = tparams[prefix_append(prefix, 'Ux')].shape[0]

    # initial/previous state
    init_state = tensor.alloc(0., n_samples, dim)

    # projected context 
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = tensor.dot(context, tparams[prefix_append(prefix,'Wc_att')]) + tparams[prefix_append(prefix,'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, h_, ctx_, alpha_, v_, pctx_,
              Wd_att, U_att, c_tt, Ux, Wx, bx, W_st, b_st):

        # attention
        pstate_ = tensor.dot(h_, Wd_att)
        pctx__ = pctx_ + pstate_[None,:,:] 
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (context * alpha[:,:,None]).sum(0) # current context

        preactx = tensor.dot(h_, Ux)
        preactx += tensor.dot(ctx_, Wx)
        preactx += bx

        h = tensor.tanh(preactx)

        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        # compute stopping probability
        ss = tensor.nnet.sigmoid(tensor.dot(h, W_st) + b_st)
        v_ = v_ * (1. - ss)[:,0][:,None]

        return h, ctx_, alpha.T, v_[:,0] #, pstate_, preact, preactx, r, u

    rval, updates = theano.scan(_step, 
                                sequences=[mask],
                                outputs_info = [init_state, 
                                                tensor.alloc(0., n_samples, context.shape[2]),
                                                tensor.alloc(0., n_samples, context.shape[0]),
                                                tensor.alloc(1., n_samples)],
                                                #None, None, None, 
                                                #None, None],
                                non_sequences=[pctx_,
                                               tparams[prefix_append(prefix,'Wd_att')],
                                               tparams[prefix_append(prefix,'U_att')],
                                               tparams[prefix_append(prefix, 'c_tt')],
                                               tparams[prefix_append(prefix, 'Ux')],
                                               tparams[prefix_append(prefix, 'Wx')],
                                               tparams[prefix_append(prefix, 'bx')],
                                               tparams[prefix_append(prefix, 'W_st')],
                                               tparams[prefix_append(prefix, 'b_st')]],
                                name=prefix_append(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile)

    rval[0] = rval[0] * rval[3][:,:,None]
    return rval


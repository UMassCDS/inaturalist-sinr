import torch
from torch.nn import functional as F
import utils

def get_loss_function(params):
    if params['loss'] == 'an_full':
        return an_full
    elif params['loss'] == 'an_slds':
        return an_slds
    elif params['loss'] == 'an_ssdl':
        return an_ssdl
    elif params['loss'] == 'an_full_me':
        return an_full_me
    elif params['loss'] == 'an_slds_me':
        return an_slds_me
    elif params['loss'] == 'an_ssdl_me':
        return an_ssdl_me
    elif params['loss'] == 'neg_log_loss':
        return neg_log_loss
    elif params['loss'] == 'bce':
        return bce
    elif params['loss'] == 'neg_log_dl_an':
        return neg_log_dl_an
    elif params['loss'] == 'bce_dl_an':
        return bce_dl_an

def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    entropy = p * neg_log(p) + (1-p) * neg_log(1-p)
    return entropy

def neg_log_loss(batch, model, params, loc_to_feats):
    """
    Calculates negative log loss of prediction at a location targeted to the corresponding annotation.

    Parameters:
        - batch: the annotation batch that supplies: locational features, locations (unused), class ids, annotation types (0 | 1)
        - model: the model used to calculate loss for
        - params: training | annotation parameters used for fine tuning
        - loc_to_feats (unused): location encoder
    
    Returns:
        - neg_log_loss: A torch Loss object that you would use to calculate backwards on.
    """

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id, types = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    types = types.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    falsy = torch.ones_like(loc_pred) - loc_pred
    truthy = loc_pred
    loc_pred = torch.where(types.unsqueeze(1) == 0, falsy, truthy)

    nl_loss = neg_log(loc_pred)
    return nl_loss.mean()

def bce(batch, model, params, loc_to_feats):
    """
    Calculates binary cross entropy loss of prediction at a location targeted to the corresponding annotation.

    Parameters:
        - batch: the annotation batch that supplies: locational features, locations (unused), class ids, annotation types (0 | 1)
        - model: the model used to calculate loss for
        - params: training | annotation parameters used for fine tuning
        - loc_to_feats (unused): location encoder
    
    Returns:
        - bce_loss: A torch Loss object that you would use to calculate backwards on.
    """

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id, types = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    types = types.to(params['device'])

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    # weights = torch.where(types == 0, torch.tensor(100.0).to(params['device']), torch.tensor(1.0).to(params['device']))
    # weights = weights.to(params['device'])
    bce_loss = F.binary_cross_entropy(loc_pred[inds[:batch_size], class_id], types.float())

    return bce_loss

def an_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))
    
    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_slds(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    
    num_classes = loc_pred.shape[1]
    bg_class = torch.randint(low=0, high=num_classes-1, size=(batch_size,), device=params['device'])
    bg_class[bg_class >= class_id[:batch_size]] += 1
    
    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class]) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_full(batch, model, params, loc_to_feats, neg_type='hard'):
    
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    # get predictions for locations and background locations
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))
    
    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log(1.0 - loc_pred) # assume negative
        loss_bg = neg_log(1.0 - loc_pred_rand) # assume negative
    elif neg_type == 'entropy':
        loss_pos = -1 * bernoulli_entropy(1.0 - loc_pred) # entropy
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand) # entropy
    else:
        raise NotImplementedError
    loss_pos[inds[:batch_size], class_id] = params['pos_weight'] * neg_log(loc_pred[inds[:batch_size], class_id])
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_full_me(batch, model, params, loc_to_feats):

    return an_full(batch, model, params, loc_to_feats, neg_type='entropy')

def an_ssdl_me(batch, model, params, loc_to_feats):
    
    return an_ssdl(batch, model, params, loc_to_feats, neg_type='entropy')

def an_slds_me(batch, model, params, loc_to_feats):
    
    return an_slds(batch, model, params, loc_to_feats, neg_type='entropy')

def neg_log_dl_an(batch, model, params, loc_to_feats, neg_type='hard'):
    """
    Loss function for fine tuning, combines neg_log loss and ssdl loss. Further discussed in fine tuning report.

    Parameters:
        - batch: the annotation batch that supplies: locational features, locations (unused), class ids, annotation types (0 | 1)
        - model: the model used to calculate loss for
        - params: training | annotation parameters used for fine tuning
        - loc_to_feats (unused): location encoder
    
    Returns:
        - loss: weighted sum of the two losses
    """
    nl_loss = neg_log_loss(batch, model, params, loc_to_feats)

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id, _ = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    loc_emb = model(rand_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_input = loc_pred[inds[:batch_size], class_id]
    dl_an = F.binary_cross_entropy(loc_pred_input, torch.zeros_like(loc_pred_input))

    return nl_loss * params['prior_weight'] + dl_an * params['dl_an_weight']


def bce_dl_an(batch, model, params, loc_to_feats, neg_type='hard'):
    """
    Loss function for fine tuning, combines bce loss and ssdl loss. Further discussed in fine tuning report.

    Parameters:
        - batch: the annotation batch that supplies: locational features, locations (unused), class ids, annotation types (0 | 1)
        - model: the model used to calculate loss for
        - params: training | annotation parameters used for fine tuning
        - loc_to_feats (unused): location encoder
    
    Returns:
        - loss: weighted sum of the two losses
    """
    bce_loss = bce(batch, model, params, loc_to_feats)

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id, _ = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)

    loc_emb = model(rand_feat, return_feats=True)
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_input = loc_pred[inds[:batch_size], class_id]
    dl_an = F.binary_cross_entropy(loc_pred_input, torch.zeros_like(loc_pred_input))

    return bce_loss * params['prior_weight'] + dl_an * params['dl_an_weight']

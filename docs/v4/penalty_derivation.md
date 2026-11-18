# What the sustainability penalty does to a predicted probability

Loss for one candidate pair (u, v), as in Eq. 12 of the manuscript, with the
BCE term and the penalty both averaged over the same set of scored pairs:

    L(q) = -[ p log q + (1 - p) log(1 - q) ] + a q,   a = lambda * risk(v)

where q = sigma(s(u, v)) is the predicted probability and p is the empirical
frequency with which the pair is labelled positive in training.

## Stationary point

    dL/dq = -p/q + (1 - p)/(1 - q) + a = 0

Multiplying by q(1 - q):

    q - p + a q (1 - q) = 0   <=>   a q^2 - (1 + a) q + p = 0

    q* = [ (1 + a) - sqrt( (1 + a)^2 - 4 a p ) ] / (2a)

(the root in [0, 1]; q* -> p as a -> 0, and q* = 0 when p = 0).

## First order in a

From q - p = -a q (1 - q): q* ~= p - a p (1 - p). Since
d logit(q)/dq = 1 / (q (1 - q)),

    logit(q*) ~= logit(p) - a = logit(p) - lambda * risk(v).

## Consequences (to be tested, not assumed)

1. To first order the training-time penalty subtracts lambda * risk(v) from
   every logit involving destination v. That is the inference-time penalty
   score - beta * risk with beta = lambda, which needs no retraining.
2. A top-k list changes only where candidates whose logits differ by less than
   lambda * |risk(v1) - risk(v2)| have different risks. If almost every
   plausible candidate has the same risk (LODHalal: 96.2% at risk 0), the
   penalty can change little; if risk varies among close competitors, it can
   change a lot. This is the formal version of the "risk concentration"
   observation.
3. The argument is pointwise. A capacity-limited model (here a dot-product
   decoder with no destination bias term) may not be able to represent a
   per-destination shift exactly, so the learned penalty can differ from the
   exact shift. scripts/34 measures the slope of (s_lambda - s_0) against
   risk(v) to check how closely trained models follow the prediction.

Checked by hand on 17 November 2026. The derivation assumes the penalty is
applied to every scored pair (penalty_on="all").

"""Exact checks for the model-derived parameter region and obstruction identity.

These checks verify the stated algebra. They do not formally verify the
upstream probability/geometric lemmas or establish literature priority.
"""
from fractions import Fraction as Q
from itertools import combinations
import json
from pathlib import Path
import sympy as s

ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/'parameters.json').read_text())
checks=[]
def margins(p):
    a,m,b,d,g,c=(Q(p[k]) for k in ('a','m','b','delta','gamma','c'))
    return {
        'dimension_grows':a,
        'period_less_than_heavy_cutoff':m-2*a,
        'global_entropy_less_than_M':m-3*a,
        'M_sublinear':1-m,
        'flip_probability_decays':b,
        'affine_global_entropy_gap':1-b-3*a,
        'affine_norm_entropy_gap':a-b,
        'light_color_margin':d/4-g,
        'long_block_color_margin':(1-d)/8-g,
        'rich_union_bound_gap':g*b/2-2*c,
        'drift_less_than_eta_gap':3-m-a,
    }
def valid(p):
    ms=margins(p)
    return all(v>=0 if k=='light_color_margin' else v>0 for k,v in ms.items())
for p in config['proposals']:
    assert valid(p),p
    checks.append({'name':p['name'],'factor_over_release':str(Q(p['c'])/Q(config['baseline_c'])),'margins':{k:str(v) for k,v in margins(p).items()}})
# Meaningful rejection controls: exact boundary must fail because logarithmic
# factors are not dominated by equal powers; zero rich exponent also fails.
p=dict(config['proposals'][-1]);p['b']=p['a'];assert not valid(p)
p=dict(config['proposals'][-1]);p['c']=str(Q(p['gamma'])*Q(p['b'])/4);assert not valid(p)
p=dict(config['proposals'][-1]);p['m']=str(3*Q(p['a']));assert not valid(p)

# Sharpened region: h_0=D*ceil(log k)^2 instead of D^2 (global entropy k^{2a}),
# (1/2-eps)-balanced outer coloring, eps_reg nonregular fraction, and light
# positions outside heavy blocks balanced separately. See sharpened-region-proof.md.
def sharpened_margins(p):
    a,m,b,d,e,er,g,c=(Q(p[k]) for k in ('a','m','b','delta','eps','eps_reg','gamma','c'))
    return {
        'dimension_grows':a,
        'short_period_cutoff_below_M':m-a,
        'global_entropy_less_than_M':m-2*a,
        'M_sublinear':1-m,
        'flip_probability_decays':b,
        'affine_global_entropy_gap':1-b-2*a,
        'affine_norm_entropy_gap':a-b,
        'balance_tolerance_positive':e,
        'balance_tolerance_below_half':Q(1,2)-e,
        'nonregular_fraction_positive':er,
        'nonregular_fraction_at_most_half':Q(1,2)-er,
        'light_threshold_positive':d,
        'outer_color_margin':(Q(1,2)-e)*(1-er)*(1-d)-g,
        'rich_union_bound_gap':g*b/2-2*c,
        'drift_less_than_eta_gap':3-m-a,
    }
def sharpened_valid(p):
    ms=sharpened_margins(p)
    return all(v>=0 if k=='nonregular_fraction_at_most_half' else v>0 for k,v in ms.items())
sharpened=[]
for p in config['sharpened_proposals']:
    assert sharpened_valid(p),p
    sharpened.append({'name':p['name'],'factor_over_release':str(Q(p['c'])/Q(config['baseline_c'])),'margins':{k:str(v) for k,v in sharpened_margins(p).items()}})
# Rejection controls: b=a loses the affine norm gap; c=gamma*b/4 gives a zero
# rich exponent; b=1-2a loses the affine global gap; gamma at the exact outer
# color bound leaves no room for the o(k) remainder block; the supremum 1/24 fails.
p=dict(config['sharpened_proposals'][-1]);p['b']=p['a'];assert not sharpened_valid(p)
p=dict(config['sharpened_proposals'][-1]);p['c']=str(Q(p['gamma'])*Q(p['b'])/4);assert not sharpened_valid(p)
p=dict(config['sharpened_proposals'][-1]);p['b']=str(1-2*Q(p['a']));assert not sharpened_valid(p)
p=dict(config['sharpened_proposals'][-1]);p['gamma']=str((Q(1,2)-Q(p['eps']))*(1-Q(p['eps_reg']))*(1-Q(p['delta'])));assert not sharpened_valid(p)
p=dict(config['sharpened_proposals'][-1]);p['c']='1/24';assert not sharpened_valid(p)
# Supremum check: with b<a, 2a+b<1 the rich gap forces c<gamma*b/4<(1/2)(1/3)/4=1/24.
assert all(Q(p['gamma'])*Q(p['b'])/4<Q(1,24) for p in config['sharpened_proposals'])

# Rank-split rich cover (rank-certificate-proof.md): the sharpened region with
# the rich condition replaced by 5c<2*gamma*b. Certificates cost min(2rho,D)
# log q (Lemmas A, B) and rank-rho progressions (2D-rho) log q (Lemma C).
def rank_margins(p):
    ms=sharpened_margins(p);del ms['rich_union_bound_gap']
    g,b,c=(Q(p[k]) for k in ('gamma','b','c'))
    ms['rank_split_rich_gap']=2*g*b/5-c
    return ms
def rank_valid(p):
    ms=rank_margins(p)
    return all(v>=0 if k=='nonregular_fraction_at_most_half' else v>0 for k,v in ms.items())
kappa=lambda x:min(2*x,Q(1));nu=lambda x:2-x
# kappa+nu is piecewise linear in x=rho/D, so its maximum is at a breakpoint.
worst=max(kappa(x)+nu(x) for x in (Q(0),Q(1,2),Q(1)))
assert worst==Q(5,2)==kappa(Q(1,2))+nu(Q(1,2))
rank=[]
for p in config['rank_certificate_proposals']:
    assert rank_valid(p),p
    g,b,c=(Q(p[k]) for k in ('gamma','b','c'))
    mu=(g*b-worst*c)/2
    for i in range(1001):
        x=Q(i,1000);theta=(kappa(x)*c+g*b-nu(x)*c)/(2*b)
        assert b*theta-kappa(x)*c>=mu>0 and b*(g-theta)-nu(x)*c>=mu
    # Outside both earlier rich conditions: c>=gamma*b/4 here, and c>gamma*b/3
    # (upstream PR #4), so the improvement is not a reparametrization.
    assert c>=g*b/4 and c>g*b/3
    rank.append({'name':p['name'],'factor_over_release':str(c/Q(config['baseline_c'])),
                 'group_margin_mu':str(mu),'margins':{k:str(v) for k,v in rank_margins(p).items()}})
# Rejection controls: b=a, b=1-2a, zero rank-split gap, gamma at the balance
# bound, and the supremum 1/15.
p=dict(config['rank_certificate_proposals'][-1]);p['b']=p['a'];assert not rank_valid(p)
p=dict(config['rank_certificate_proposals'][-1]);p['b']=str(1-2*Q(p['a']));assert not rank_valid(p)
p=dict(config['rank_certificate_proposals'][-1]);p['c']=str(2*Q(p['gamma'])*Q(p['b'])/5);assert not rank_valid(p)
p=dict(config['rank_certificate_proposals'][-1]);p['gamma']=str((Q(1,2)-Q(p['eps']))*(1-Q(p['eps_reg']))*(1-Q(p['delta'])));assert not rank_valid(p)
p=dict(config['rank_certificate_proposals'][-1]);p['c']='1/15';assert not rank_valid(p)
assert all(2*Q(p['gamma'])*Q(p['b'])/5<Q(1,15) for p in config['rank_certificate_proposals'])

# Symbolic identities (1)-(4) and Lemma A in dimension 3, with generic vectors.
dot=lambda u,v:sum(i*j for i,j in zip(u,v))
Y0,tv,be,dm,A0,Bv,Cv=(s.symbols(n+'0:3') for n in ('Y','t','beta','dm','A','B','C'))
ca,sa,s0,sv,qs,jv,j1,j2=s.symbols('c_a sigma_a sigma_0 s q j j_1 j_2')
w=dot(be,be);W=qs**2*w
Ya=[Y0[i]+ca*(tv[i]+be[i])-dm[i] for i in range(3)]
za=qs**2*(2*dot(Ya,be)+sa*w);z0=qs**2*(2*dot(Y0,be)+s0*w)
Lam=(qs**2*dot(Y0,Y0)+2*ca*qs**2*dot(Y0,tv)-(ca**2+ca*sa)*W-2*qs**2*dot(Y0,dm)
     +qs**2*(ca**2*dot(tv,tv)+dot(dm,dm)-2*ca*dot(tv,dm)))
Yj=[Ya[i]+sv*be[i] for i in range(3)]
assert s.expand(qs**2*dot(Ya,Ya)-Lam-ca*za)==0                       # Lemma A
assert s.expand(qs**2*dot(Yj,Yj)-qs**2*dot(Ya,Ya)-sv*(sv-sa)*W-sv*za)==0  # (2)
assert s.expand(dot(dm,be)-ca*(dot(tv,be)+w)-(sa-s0)*w/2-(z0-za)/(2*qs**2))==0  # (4)
fj=lambda j:qs**2*dot([A0[i]+Cv[i]+j*Bv[i] for i in range(3)],[A0[i]+Cv[i]+j*Bv[i] for i in range(3)])
eta=(fj(j2)-fj(j1))/(j2-j1)
assert s.simplify(fj(jv)-qs**2*dot([A0[i]+Cv[i] for i in range(3)],[A0[i]+Cv[i] for i in range(3)])
                  -jv*(jv-j1-j2)*qs**2*dot(Bv,Bv)-jv*eta)==0             # (3)

# Lemma C on exact instances: beta_i=sum_{l<i} e_l q^{l-i} is the centered
# representative of n'/q^i, and N1=<Dm,(e_{i-1})> is the integer nearest X/q
# whenever |X-q^2<Dm,beta>|<3, with |Dm_i|<=h+1 and 32D(k+2)<k^2.
import random
rng=random.Random(2026);digit_cases=0
def centered(x):
    x-=x.numerator//x.denominator
    return x-1 if x>=Q(1,2) else x
while digit_cases<2000:
    qq=rng.choice((1009,10007,100003));D=rng.randint(2,6);k=rng.randint(8,80);h=rng.randint(1,k-1)
    if 32*D*(k+2)>=k*k:continue
    lim=int(2*qq/(k*k));e=[rng.randint(-lim,lim) for _ in range(D)]
    npr=sum(e[l]*qq**l for l in range(D))
    beta=[sum(Q(e[l],qq**(i+1-l)) for l in range(i+1)) for i in range(D)]
    if any(abs(x)>Q(2,k*k) for x in beta):continue
    assert all(beta[i]==centered(Q(npr,qq**(i+1))) for i in range(D))
    dmv=[rng.randint(-(h+1),h+1) for _ in range(D)]
    X=qq*qq*dot(dmv,beta)+Q(rng.randint(-2999,2999),1000)
    assert round(X/qq)==dot(dmv,e)
    digit_cases+=1

# Independent finite tests of the exact half-open norm-band lemma. The proof
# is analytic; these tests include unit-distance equality and t=0 cases.
points=[Q(i,4) for i in range(-20,21)];band_cases=0
for ts in combinations(points,3):
    if min(ts[i+1]-ts[i] for i in range(2))<1:continue
    for shift in (Q(0),Q(1,4),Q(1,2),Q(3,4)):
        bands=[(t*t+shift).__floor__() for t in ts]
        assert len(set(bands))>1
        band_cases+=1

# Symbolic proof identities for the matrix-profile obstruction.
r,x,A=s.symbols('r x A',positive=True)
assert s.expand((r-1)**3*(r+1))==s.expand(r**4-2*r**3+2*r-1)
f=(x/2)**s.Rational(1,3)*(2*A+x)/6
assert s.simplify(9*s.diff(f,x,2)-2**s.Rational(2,3)*(x-A)/(3*x**s.Rational(5,3)))==0
# Exact cube comparison avoids numerical cube roots for tripling.
def profile_cube(a,b):
    a,b=sorted((a,b));return Q(3*a-1,2)*Q(2*b+a-1,2)**3
tripling_cases=0
for a in range(1,101):
    assert profile_cube(1,a)==a**3
    assert profile_cube(a,a)==Q(3*a-1,2)**4
    for h in range(1,101):
        assert profile_cube(a,3*h+a-1)>=27*profile_cube(a,h)
        assert profile_cube(a,h)<=Q(a+h-1)**4
        tripling_cases+=1

out={'scope':'Exact algebra and finite checks only; not an independent formal proof of upstream results',
     'parameter_certificates':checks,'sharpened_parameter_certificates':sharpened,
     'rank_certificate_parameter_certificates':rank,
     'rank_split':{'worst_cost_sum':str(worst),'worst_rank_fraction':'1/2','coefficient_supremum':'1/15','digit_test_cases':digit_cases},
     'negative_controls_rejected':13,
     'half_open_band_test_cases':band_cases,'profile_exact_test_cases':tripling_cases,
     'symbolic_identities':'passed'}
(ROOT/'experiments'/'exact-audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

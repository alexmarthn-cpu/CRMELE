"""Synthesize the music bed and SFX for the ElevateHub CRM video (deterministic)."""
import numpy as np, soundfile as sf
SR = 44100
rng = np.random.default_rng(7)

def env(n, a, r):
    e = np.ones(n); a = int(a*SR); r = int(r*SR)
    if a: e[:a] = np.linspace(0, 1, a)
    if r: e[-r:] *= np.linspace(1, 0, r)
    return e

def note(f, dur, amp=1.0, harm=(1, .5, .25), a=.01, r=.3):
    t = np.arange(int(dur*SR))/SR
    s = sum(h*np.sin(2*np.pi*f*(i+1)*t) for i, h in enumerate(harm))
    return amp*s*env(len(t), a, r)

def save(name, x, stereo=True):
    x = x/ (np.max(np.abs(x))+1e-9) * 0.89
    if stereo and x.ndim == 1: x = np.stack([x, x], 1)
    sf.write(name, x, SR)

# --- notification ping (two-tone, glassy)
p = np.zeros(int(.6*SR))
a = note(1318.5, .35, 1, (1, .2), .003, .3); b = note(1760, .4, .8, (1, .15), .003, .35)
p[:len(a)] += a; o = int(.09*SR); p[o:o+len(b)] += b
save("assets/sfx/ping.wav", p)

# --- whoosh (filtered noise sweep)
n = int(1.0*SR); w = rng.standard_normal(n)
t = np.arange(n)/SR
cut = np.interp(t, [0, .55, 1.0], [300, 5000, 600])
y = np.zeros(n); lp = 0
alpha = 2*np.pi*cut/SR/(1+2*np.pi*cut/SR)
for i in range(n):
    lp += alpha[i]*(w[i]-lp); y[i] = lp
y *= np.sin(np.pi*np.clip(t/1.0, 0, 1))**2
save("assets/sfx/whoosh.wav", y)

# --- soft click / pop
c = note(900, .08, 1, (1, .3), .001, .07) + .3*rng.standard_normal(int(.08*SR))*env(int(.08*SR), 0, .075)
save("assets/sfx/click.wav", c)

# --- success chime (major arpeggio)
ch = np.zeros(int(1.6*SR))
for k, f in enumerate([783.99, 987.77, 1174.66, 1567.98]):
    s = note(f, 1.0, .8, (1, .3, .1), .005, .9); o = int(k*.08*SR); ch[o:o+len(s)] += s
save("assets/sfx/chime.wav", ch)

# --- logo impact (sub boom + shimmer)
n = int(3.5*SR); t = np.arange(n)/SR
boom = np.sin(2*np.pi*(55*t - 20*t**2*0))*np.exp(-t*2.2)
shim = sum(np.sin(2*np.pi*f*t)*np.exp(-t*1.3)*.18 for f in (1046.5, 1568, 2093, 2637))
save("assets/sfx/impact.wav", boom + shim + .15*note(261.63, 3.5, 1, (1, .5, .3), .01, 2.5))

# --- music bed: 40s ambient pad, growing, progression Am F C G (in Hz roots)
dur = 40.0; n = int(dur*SR); t = np.arange(n)/SR
mix = np.zeros((n, 2))
chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [261.63, 329.63, 392.0], [196.0, 246.94, 293.66]]
bar = 2.4  # seconds per chord
growth = np.interp(t, [0, 5, 22, 33.5, 36, 40], [.25, .45, .75, 1.0, .9, .0])
for k in range(int(dur/bar)+1):
    st = k*bar; ch = chords[k % 4]
    seg = int((bar+1.2)*SR); tt = np.arange(seg)/SR
    e = env(seg, .8, 1.4)
    for j, f in enumerate(ch):
        det = [0.997, 1.003]
        for side in (0, 1):
            s = np.sin(2*np.pi*f*det[side]*tt) + .35*np.sin(2*np.pi*2*f*det[side]*tt)
            i0 = int(st*SR); i1 = min(n, i0+seg)
            mix[i0:i1, side] += .12*s[:i1-i0]*e[:i1-i0]
    # bass
    b = note(ch[0]/2, bar, .35, (1, .4), .02, .6); i0 = int(st*SR); i1 = min(n, i0+len(b))
    mix[i0:i1] += b[:i1-i0, None]
# pulse (eighth-note plucks) entering from 5s, stronger over time
step = bar/8
for k in range(int(dur/step)):
    st = k*step
    if st < 5.0 or st > 36.5: continue
    ch = chords[int(st//bar) % 4]; f = ch[k % 3]*2
    s = note(f, .25, .10, (1, .5, .2), .002, .22); i0 = int(st*SR); i1 = min(n, i0+len(s))
    pan = .5 + .3*np.sin(k)
    mix[i0:i1, 0] += s[:i1-i0]*(1-pan); mix[i0:i1, 1] += s[:i1-i0]*pan
# soft kick from 22s
for k in range(int(dur/(bar/2))):
    st = k*bar/2
    if st < 21.9 or st > 33.6: continue
    kk = np.sin(2*np.pi*(60*np.arange(int(.3*SR))/SR))*np.exp(-np.arange(int(.3*SR))/SR*14)*.5
    i0 = int(st*SR); mix[i0:i0+len(kk)] += kk[:, None]
mix *= growth[:, None]
# simple feedback delay reverb-ish
d = int(.31*SR); out = mix.copy()
for _ in range(4): out[d:] += .35*out[:-d]
out = out/np.max(np.abs(out))*.8
sf.write("assets/music/bed.wav", out, SR)
print("ok")

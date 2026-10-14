"""Unrestarted Arnoldi/GMRES written only with finite Program algebra.

The action callback is elaborated while the enclosing GraphFunction is authored;
its equations become the same native graph. No callable is retained at runtime.
"""
from __future__ import annotations
import pops.math as math


def gmres(p, rhs, guess, action, *, atol, capacity):
    def amap(b, a, fn): return b.map(b.domain(a), lambda c, i: fn(c, i))
    def axpy(b, a, s, v): return amap(b, a, lambda c, i: c.at(a, i)+s*c.at(v, i))
    def dot(b, a, v):
        return b.reduce(amap(b, a, lambda c, i:c.at(a, i)*c.at(v, i)), b.float64(0),
                        lambda c, total, x:total+x, order='ascending')
    def get(b, a, j): return b.at(a, b.index(b.domain(a), j))
    def put(b, a, j, value):
        return amap(b, a, lambda c, i:c.branch(c.index_value(i)==j,
            lambda z:value, lambda z:z.at(a, i)))
    n = p.size(p.domain(rhs))
    columns = p.range_set(n, capacity=capacity, ownership='replicated')
    rows = p.range_set(n+1, capacity=capacity+1, ownership='replicated')
    entries = p.range_set(n*(n+1), capacity=capacity*(capacity+1), ownership='replicated')
    residual = axpy(p, rhs, -1, action(p, guess))
    norm = math.sqrt(dot(p, residual, residual))
    zero = amap(p, rhs, lambda b, i:b.float64(0))
    first = p.branch(norm > 0, lambda b:amap(b, residual, lambda c, i:c.at(residual, i)/norm),
                     lambda b:zero)
    basis = p.map(rows, lambda b, i:b.branch(b.index_value(i)==0, lambda c:first, lambda c:zero))
    hessenberg = p.map(entries, lambda b, i:b.float64(0))
    rotations = p.map(columns, lambda b, i:b.float64(0))
    g = p.map(rows, lambda b, i:b.branch(b.index_value(i)==0, lambda c:norm, lambda c:c.float64(0)))
    initial = p.product(basis=basis, H=hessenberg, c=rotations, s=rotations, g=g,
                        k=p.int64(0), defect=norm*norm)
    def iteration(b, state):
        k = state['k']
        w = action(b, get(b, state['basis'], k))
        def orthogonalize(c, carry, j):
            v = get(c, state['basis'], j)
            hij = dot(c, v, carry['w'])
            return c.product(w=axpy(c, carry['w'], -hij, v), H=put(c, carry['H'], j*n+k, hij))
        orthogonal = b.fold(b.product(w=w, H=state['H']), k+1, orthogonalize, max_steps=capacity)
        beta = math.sqrt(dot(b, orthogonal['w'], orthogonal['w']))
        H = put(b, orthogonal['H'], (k+1)*n+k, beta)
        next_vector = b.branch(beta > 0,
            lambda c:amap(c, orthogonal['w'], lambda d, i:d.at(orthogonal['w'], i)/beta),
            lambda c:zero)
        V = put(b, state['basis'], k+1, next_vector)
        def rotate(c, matrix, j):
            upper, lower = get(c, matrix, j*n+k), get(c, matrix, (j+1)*n+k)
            cosine, sine = get(c, state['c'], j), get(c, state['s'], j)
            matrix = put(c, matrix, j*n+k, cosine*upper+sine*lower)
            return put(c, matrix, (j+1)*n+k, -sine*upper+cosine*lower)
        H = b.fold(H, k, rotate, max_steps=capacity)
        upper, lower = get(b, H, k*n+k), get(b, H, (k+1)*n+k)
        denominator = math.sqrt(upper*upper+lower*lower)
        cosine, sine = upper/denominator, lower/denominator
        H = put(b, put(b, H, k*n+k, denominator), (k+1)*n+k, b.float64(0))
        gk = get(b, state['g'], k)
        g = put(b, put(b, state['g'], k, cosine*gk), k+1, -sine*gk)
        return b.product(basis=V, H=H, c=put(b, state['c'], k, cosine),
                         s=put(b, state['s'], k, sine), g=g, k=k+1, defect=(sine*gk)**2)
    outcome = p.require_success(p.iterate(initial, iteration,
        lambda b, state:state['defect'] <= atol, max_steps=n))
    final = outcome['state']; k = final['k']
    coefficients = p.map(columns, lambda b, i:b.float64(0))
    def back_substitute(b, y, offset):
        i = k-1-offset
        tail = b.fold(b.float64(0), k-i-1, lambda c, total, j:
            total+get(c, final['H'], i*n+(i+1+j))*get(c, y, i+1+j), max_steps=capacity)
        yi = (get(b, final['g'], i)-tail)/get(b, final['H'], i*n+i)
        return put(b, y, i, yi)
    coefficients = p.fold(coefficients, k, back_substitute, max_steps=capacity)
    answer = p.fold(guess, k, lambda b, x, j:axpy(b, x, get(b, coefficients, j),
                                                get(b, final['basis'], j)), max_steps=capacity)
    return answer, outcome['iterations']

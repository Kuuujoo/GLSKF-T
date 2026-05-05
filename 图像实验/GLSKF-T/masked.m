function [U, V, S] = masked(G, W, U, V, S, invK1, invK2, rho, cgN, tol)
% masked 只用观测位置更新 t-SVD 全局因子

U = updU(G, W, U, V, S, invK1, rho, cgN, tol);
V = updV(G, W, U, V, S, invK2, rho, cgN, tol);
S = updS(G, W, U, V, S, cgN, tol);
end

function U2 = updU(G, W, U, V, S, K, rho, cgN, tol)
b = tproduct(tproduct(W .* G, V), ttranspose(S));
A = @(x) apU(x, W, V, S, K, rho, size(U));
U2 = reshape(cg_solve(A, b(:), U(:), cgN, tol), size(U));
end

function y = apU(x, W, V, S, K, rho, sz)
U = reshape(x, sz);
M = tproduct(tproduct(U, S), ttranspose(V));
g = tproduct(tproduct(W .* M, V), ttranspose(S));
r = mode1(K, U);
y = g(:) + rho * r(:);
end

function V2 = updV(G, W, U, V, S, K, rho, cgN, tol)
b = tproduct(tproduct(ttranspose(W .* G), U), S);
A = @(x) apV(x, W, U, S, K, rho, size(V));
V2 = reshape(cg_solve(A, b(:), V(:), cgN, tol), size(V));
end

function y = apV(x, W, U, S, K, rho, sz)
V = reshape(x, sz);
M = tproduct(tproduct(U, S), ttranspose(V));
g = tproduct(tproduct(ttranspose(W .* M), U), S);
r = mode1(K, V);
y = g(:) + rho * r(:);
end

function S2 = updS(G, W, U, V, S, cgN, tol)
lam = 15;
b = tproduct(tproduct(ttranspose(U), W .* G), V);
A = @(x) apS(x, W, U, V, lam, size(S));
S2 = reshape(cg_solve(A, b(:), S(:), cgN, tol), size(S));
end

function y = apS(x, W, U, V, lam, sz)
S = reshape(x, sz);
M = tproduct(tproduct(U, S), ttranspose(V));
g = tproduct(tproduct(ttranspose(U), W .* M), V);
y = g(:) + lam * x;
end

function Y = mode1(K, X)
Y = zeros(size(X), 'like', X);
for k = 1:size(X, 3)
    Y(:, :, k) = K * X(:, :, k);
end
end

function x = cg_solve(A, b, x0, cgN, tol)
x = x0;
r = b - A(x);
p = r;
rr = real(r' * r);
if sqrt(max(rr, 0)) < tol
    return;
end

for i = 1:cgN
    Ap = A(p);
    pAp = real(p' * Ap);
    if abs(pAp) < 1e-14
        break;
    end

    a = rr / pAp;
    x = x + a * p;
    r = r - a * Ap;
    rr2 = real(r' * r);
    if sqrt(max(rr2, 0)) < tol
        break;
    end

    p = r + (rr2 / rr) * p;
    rr = rr2;
end
end

function res = rungraph(D, N, h, nsamples, take_map)

rng default;

if ~exist('N', 'var') || isempty(N)
    N = 40; % participants
end
if ~exist('h', 'var')
    h = init_hyperparams;
end
if ~exist('nsamples', 'var')
    nsamples = 10000;
end
if ~exist('take_map', 'var')
    take_map = false;
end

tic;

for s = 1:N % for each simulated subject
    if take_map
        [H, P] = sample_c(D, h, nsamples);
        [~,I] = max(P); % MAP H
        H = H(I);
    else
        [H, P] = sample_c(D, h, 1, nsamples);
    end
    chosen_H{s} = H;
end

toc;

h = init_hyperparams;

for s = 1:N % for each simulated subject
    %fprintf('pick bus stop subject %d\n', s);

    H = chosen_H{s};

    H = populate_H(H, D); % fill up bridges

    b = zeros(1,D.G.N);
    for i = 1:length(H.cnt)
        for j = i+1:length(H.cnt)
            bridge = H.b{i,j};
            if ~isempty(bridge)
                b(bridge(1)) = 1;
                b(bridge(2)) = 1;
            end
        end
    end

    b = find(b);
    if isempty(b)
        % only 1 cluster -> all nodes are fine
        b = 1:D.G.N;
    end

    loc(s,:) = datasample(b, 3); % pick 3 at random (with replacement)

    % eps-greedy: choose random node w/ small prob
    if rand() < 1 - h.eps
        loc(s,:) = datasample(1:D.G.N, 3);
    end
end

res.h = h;
res.loc = loc;
res.chosen_H = chosen_H;
res.D = D;

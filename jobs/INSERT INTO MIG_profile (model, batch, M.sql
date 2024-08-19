UPDATE MIG_profile
SET latency = 0.045
WHERE model = 'embedding' AND batch = 1280 AND mig_partition = '1g.10gb';
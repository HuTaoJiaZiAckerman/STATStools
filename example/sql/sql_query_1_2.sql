SELECT
    CAST(chra AS UNSIGNED) AS chra,
    windowa,
    trait_id,
    CAST(allelea AS UNSIGNED) AS allelea,
    CAST(allelea_mutant_paternal AS UNSIGNED) AS allelea_mutant_paternal,
    trait_male_diff_paternal
FROM
    t_mutation_effect_all_info_chr1
WHERE
    trait_id = 2

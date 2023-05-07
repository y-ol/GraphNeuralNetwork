# Experiment Plan

## Experiment-Dateibaum:

experiment_results:

-   dataset_A
    -   hyperparams.json
    -   hpconfig_0
        -   repeat_0.json
        -   repeat_1.json
        -   repeat_2.json
    -   hpconfig_1
        -   repeat_0.json
        -   ...
    -   hpconfig_2
    -   ...
-   dataset_B
    -   hyperparams.json
    -   hpconfig_0
        -   repeat_0.jon
        -   ...
    -   ...
-   ...

## Schritte

1. Für jedes Dataset JSON-Datei mit Liste von Hyperparameterkonfigurationen erzeugen (hyperparams.json).
   Jeder Hyperparameter-Konfig. eine eindeutige ID geben (numerisch reicht)
2. Modelle trainieren (für ein bestimmtes Dataset):
    1. Hyperparam-JSON laden und dessen Einträge durchlaufen
    2. Für jeden Eintrag i und jedes j in {0,1,2} prüfen, ob bereits eine Datei hpconfig_i/repeat_j.json existiert. Falls ja, skip.
    3. Falls ein hpconfig_i/repeat_j.json fehlt, ein Modell für HP-Config i erzeugen.
    4. Erzeugtes Modell auf dem Dataset trainieren (mit Early-Stopping gemäß Validation-Error und restore_best_weights=True)
    5. Training + Val + Test Metriken (inkl. Loss) in einem Dict/Liste speichern (z.B. mit den Keys: train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy). (Optional, nice-to-have: Die Keras-History (Return-Wert von model.fit) in eine Liste umwandeln und auch in das Ergebnis-Dict packen).
    6. Das Ergebnis-Dict in der Datei hpconfig_i/repeat_j.json abspeichern (json.dump)
3. Ergebnisse aggregieren:
    1.

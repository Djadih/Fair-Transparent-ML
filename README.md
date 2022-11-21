

## Procedure

### Setup

Run the experiment Python script with `python3 run_experiment.py`. Make sure that you have the correct conda environment. Immediately, the script will prompt you to enter the name of the participant. Once you enter the name, the script will create a log file in the `session_results` folder named `{participantName}_log.csv`.

Next, create the refreshable Excel sheet that participants can use to see their custom model inputs and model output.
1. Create a new blank Excel workbook.
2. Under the "Data" tab, click "From Text". This will prompt you to locate the log file for the participant, which is the `session_results/{participantName}_log.csv` file mentioned above.
3. Next, the Text Import Wizard will show up. Under "Original data type", choose "<ins>D</ins>elimited" and press "<ins>N</ins>ext >". Then, under "Delimiters", check the "<ins>C</ins>omma" box and press "<ins>F</ins>inish" in the bottom right.
4. Then, the "Import Data" wizard will appear. Press the "<ins>P</ins>roperties..." box in the bottom left corner. Under "Refresh control", uncheck the "Pro<ins>m</ins>pt for file name on refresh" box and press "OK" button on the bottom. Then press "OK" on the Import Data wizard to finish.

The reason we're doing this is so users can refresh their log by pressing the "Refresh" button under the "Data" tab.

### Participant Activity

Now, once the script has finished training all the models, it will prompt you to enter the model to use. The following table describes which model goes with which model alias.

| Model Alias | Model Type                               |
| ----------- | -----------                              |
| Albatross   | Adversarial Debiasing                    |
| Beaver      | Plain                                    |
| Chamelon    | Calibrated Equalized Odds Post-Processing|
| Dragonfly   | GerryFairClassifier                      |

When the participant decides what they want the datapoint to be, simply follow the script's prompts to enter their information.


## Miscellaneous Tips
Since we're using an Excel spreadsheet to represent a refreshable log, add filters by pressing the "Filter" button under the "Data" tab to allow participants to sort through the various tabular data types.
The script already sorts all data by model type, but if participants want to sort by another column, they are able to do so.

Participants can refresh their log by pressing the "Refresh" button under the "Data" tab.

Other Excel tips:
Use conditional formatting on the "Predicted Income..." column to better view the model's outputs. Choose colors that make sense (i.e. red for "LESS" and blue/green for "MORE")



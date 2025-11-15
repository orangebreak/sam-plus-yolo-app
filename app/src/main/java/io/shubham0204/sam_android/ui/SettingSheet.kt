package io.shubham0204.sam_android.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun SettingsSheet(
    // Pass the viewModel instance from the parent
    viewModel: SharedViewModel = viewModel()
) {
    // Observe the state from the ViewModel
    val settings by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
    ) {
        Text("Confidence Threshold", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))

        // Display the current threshold value
        Text(
            text = String.format("%.2f", settings.confidenceThreshold),
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyLarge
        )

        // The Slider to control the value
        Slider(
            value = settings.confidenceThreshold,
            onValueChange = { newValue ->
                // When the slider is moved, update the ViewModel
                viewModel.setConfidenceThreshold(newValue)
            },
            valueRange = 0.0f..1.0f, // Sensible range for confidence
            // steps = 7 // Creates 8 steps (0.1, 0.2, ..., 0.9)

            modifier = Modifier.padding(horizontal = 24.dp)
        )
    }
}

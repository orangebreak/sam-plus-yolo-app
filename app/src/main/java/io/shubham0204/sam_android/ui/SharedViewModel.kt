package io.shubham0204.sam_android.ui

// In a new file, e.g., /ui/SharedViewModel.kt
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

// Data class to hold all settings
data class DetectorSettings(
    val confidenceThreshold: Float = 0.5f
    // You can add other settings here later, like iouThreshold, etc.
)

class SharedViewModel : ViewModel() {

    // Backing property to hold the mutable state
    private val _uiState = MutableStateFlow(DetectorSettings())

    // Publicly exposed state flow that is read-only
    val uiState = _uiState.asStateFlow()

    /**
     * Updates the confidence threshold.
     * @param threshold The new value from the slider, between 0.0 and 1.0.
     */
    fun setConfidenceThreshold(threshold: Float) {
        _uiState.update { currentState ->
            currentState.copy(confidenceThreshold = threshold)
        }
    }
}

package com.dllm.mesh

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.dllm.mesh.ui.ChatScreen
import com.dllm.mesh.ui.DevicesScreen
import com.dllm.mesh.ui.PairingScreen
import com.dllm.mesh.ui.SettingsScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DllmMeshTheme {
                DllmMeshApp()
            }
        }
    }
}

/** LAN-signal theme: ink panels, single teal accent, amber reserved for warnings. */
@Composable
fun DllmMeshTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = Color(0xFF101418),
            surface = Color(0xFF171D24),
            primary = Color(0xFF2DD4BF),
            secondary = Color(0xFF93A1B0),
            tertiary = Color(0xFFF5B544),
            onBackground = Color(0xFFE8EDF2),
            onSurface = Color(0xFFE8EDF2),
        ),
        content = content,
    )
}

@Composable
fun DllmMeshApp() {
    val navController = rememberNavController()
    val backStack by navController.currentBackStackEntryAsState()
    val route = backStack?.destination?.route ?: "chat"

    Scaffold(
        containerColor = Color(0xFF101418),
        bottomBar = {
            NavigationBar(containerColor = Color(0xFF171D24)) {
                NavigationBarItem(
                    selected = route == "chat",
                    onClick = {
                        if (route != "chat") {
                            navController.navigate("chat") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.Chat, contentDescription = null) },
                    label = { Text("Chat") },
                )
                NavigationBarItem(
                    selected = route == "devices",
                    onClick = {
                        if (route != "devices") {
                            navController.navigate("devices") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.List, contentDescription = null) },
                    label = { Text("Devices") },
                )
                NavigationBarItem(
                    selected = route == "pairing",
                    onClick = {
                        if (route != "pairing") {
                            navController.navigate("pairing") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.QrCodeScanner, contentDescription = null) },
                    label = { Text("Pairing") },
                )
                NavigationBarItem(
                    selected = route == "settings",
                    onClick = {
                        if (route != "settings") {
                            navController.navigate("settings") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text("Settings") },
                )
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = "chat",
            modifier = Modifier.padding(padding),
        ) {
            composable("chat") { ChatScreen() }
            composable("devices") { DevicesScreen() }
            composable("pairing") { PairingScreen() }
            composable("settings") { SettingsScreen() }
        }
    }
}

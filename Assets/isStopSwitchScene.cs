using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class isStopSwitchScene : MonoBehaviour
{
    public string sceneName = "";

    public void Update()
    {
        if (GetComponent<Rigidbody>().IsSleeping() && Time.time > 8)
        {
            if (sceneName != "")
            {
                SceneManager.LoadScene(sceneName);
            }
            else
            {
                int nextIndex = SceneManager.GetActiveScene().buildIndex + 1;
                if (nextIndex < SceneManager.sceneCountInBuildSettings)
                {
                    SceneManager.LoadScene(nextIndex);
                }
                else
                {
                    SceneManager.LoadScene(0);
                }
            }
        }
    }
}
